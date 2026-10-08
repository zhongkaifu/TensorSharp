// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Qwen4Exp's optional CUDA expert tier. GGUF bytes remain quantized and are
// copied into compact, persistent slots; ordinary upstream ggml CUDA kernels
// compute the complete routed FFN. No CPU/GPU partial-sum split is involved.
#include "ggml_ops_internal.h"
#include "ggml_ops_precision_policy.h"
#include "ggml_ops_shared_cache_budget.h"
#include "ggml-impl.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cerrno>
#include <cstdlib>
#include <list>

namespace tsg
{
    namespace
    {
        constexpr std::size_t kMiB = std::size_t(1) << 20;
        // This allowance covers small upstream CUDA work buffers in addition
        // to every byte the graph allocator reports. CUDA's shared backend
        // pool and driver capture storage also need physical-device headroom;
        // their total cannot be inferred from the graph allocator.
        constexpr std::size_t kWorkspaceAllowance = kMiB;
        constexpr std::size_t kDeviceHeadroom = std::size_t(512) << 20;
        constexpr int kMaxUsed = 64;

        std::mutex g_cache_mutex;
        std::size_t g_reserved = 0;
        std::uint64_t g_hits = 0, g_misses = 0, g_calls = 0;
#if defined(TSG_GGML_TEST_HOOKS)
        std::atomic<int> g_test_failure_stage{0};
        std::atomic<std::size_t> g_test_physical_bytes{0};
        bool test_failure(int stage)
        {
            int expected = stage;
            return g_test_failure_stage.compare_exchange_strong(expected, 0);
        }
#endif

        std::size_t read_positive_size(const char* name, std::size_t fallback)
        {
            const char* value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') return fallback;
            // Match the managed placement parser: unsigned decimal digits,
            // with no sign or surrounding whitespace.
            const char* digits = value;
            if (digits[0] < '0' || digits[0] > '9') return 0;
            for (const char* p = digits; *p != '\0'; ++p)
                if (*p < '0' || *p > '9') return 0;
            errno = 0;
            char* end = nullptr;
            const unsigned long long n = std::strtoull(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0'
                || n > std::numeric_limits<std::size_t>::max()) return 0;
            return static_cast<std::size_t>(n);
        }

        std::size_t cache_budget()
        {
            static const std::size_t bytes = [] {
                const std::size_t mb = read_positive_size("TS_HOST_MOE_EXPERT_CACHE_MB", 0);
                const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
                return mb <= maximum / kMiB ? mb * kMiB : 0;
            }();
            return bytes;
        }

        std::size_t cache_layer_count()
        {
            // The opt-in feature is restricted to Qwen4Exp. Its production
            // checkpoint has 48 layers; smaller fixtures can set this value.
            static const std::size_t layers = read_positive_size("TS_HOST_MOE_EXPERT_CACHE_LAYERS", 48);
            return layers;
        }

        bool diagnostics()
        {
            static const bool enabled = [] {
                const char* e = std::getenv("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS");
                return e != nullptr && e[0] == '1';
            }();
            return enabled;
        }

        struct CacheEntry
        {
            ggml_backend_t backend = nullptr;
            int layer = -1, hidden = 0, n_ff = 0, n_used = 0, num_experts = 0;
            const void* source[3] = {};
            int type[3] = {};
            std::size_t stride[3] = {};
            std::size_t bytes = 0;
            int capacity = 0;
            std::vector<int> expert_for_slot;
            std::vector<std::uint64_t> last_used;
            std::uint64_t clock = 0;
            ggml_context* ctx = nullptr;
            ggml_gallocr_t allocator = nullptr;
            ggml_cgraph* graph = nullptr;
            ggml_tensor* weight[3] = {};
            ggml_tensor* input = nullptr;
            ggml_tensor* ids = nullptr;
            ggml_tensor* routes = nullptr;
            ggml_tensor* output = nullptr;
            // The complete compact graph (weights, routing/activation payload,
            // padding and the existing workspace allowance) is one kind-1 owner.
            // This allocator deliberately does not use the kind-2 graph wrapper.
            std::shared_ptr<SharedCacheCharge> shared_charge;
#if defined(TSG_GGML_TEST_HOOKS)
            std::size_t test_physical_bytes = 0;
#endif

            ~CacheEntry()
            {
                // The last direct output copy can still be queued after the
                // helper returns. Drain it before eviction/invalidation frees
                // any source storage; CUDA remains alive at both release hooks.
                if (allocator != nullptr)
                {
                    sync_backend(backend);
                    ggml_gallocr_free(allocator);
                    allocator = nullptr;
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_physical_bytes.fetch_sub(test_physical_bytes);
                    test_physical_bytes = 0;
#endif
                }
                if (ctx != nullptr) ggml_free(ctx);
                // ggml's free is void/fatal on unrecoverable CUDA errors. Never
                // refund before it returns, including allocation/commit rollback.
                shared_charge.reset();
            }

            bool matches(const HostMoeSegment& hm, ggml_backend_t b) const
            {
                return backend == b && layer == hm.layer && hidden == hm.hidden
                    && n_ff == hm.n_ff && n_used == hm.n_used && num_experts == hm.num_experts
                    && source[0] == hm.gate_data && source[1] == hm.up_data && source[2] == hm.down_data
                    && type[0] == hm.gate_type && type[1] == hm.up_type && type[2] == hm.down_type;
            }
        };

        // LRU entries bound multiple loaded model identities as well as the
        // per-layer slots. Each layer has at most budget / layer_count bytes,
        // preventing early layers from consuming every slot on the first token.
        std::list<std::unique_ptr<CacheEntry>> g_entries;

#ifdef TSG_GGML_USE_CUDA
        bool valid_layout(const HostMoeSegment& hm, std::size_t (&stride)[3])
        {
            if (hm.seq_len <= 0 || hm.seq_len > TSG_PRECISION_DECODE_COLUMNS
                || hm.activation != 0 || hm.tp_reduced != 0
                || hm.layer < 0 || hm.hidden <= 0 || hm.n_ff <= 0
                || hm.n_used <= 0 || hm.n_used > kMaxUsed || hm.n_used > hm.num_experts
                || hm.gate_data == nullptr || hm.up_data == nullptr || hm.down_data == nullptr
                || hm.gate_bias != nullptr || hm.up_bias != nullptr || hm.down_bias != nullptr
                || hm.gate_ne0 != hm.hidden || hm.up_ne0 != hm.hidden
                || hm.gate_ne1 != hm.n_ff || hm.up_ne1 != hm.n_ff
                || hm.down_ne0 != hm.n_ff || hm.down_ne1 != hm.hidden)
                return false;
            const int types[3] = {hm.gate_type, hm.up_type, hm.down_type};
            const int widths[3] = {hm.hidden, hm.hidden, hm.n_ff};
            const int rows[3] = {hm.n_ff, hm.n_ff, hm.hidden};
            const std::int64_t totals[3] = {hm.gate_bytes, hm.up_bytes, hm.down_bytes};
            for (int i = 0; i < 3; ++i)
            {
                if (types[i] < 0 || types[i] >= GGML_TYPE_COUNT
                    || !ggml_is_quantized(static_cast<ggml_type>(types[i]))) return false;
                const int64_t block = ggml_blck_size(static_cast<ggml_type>(types[i]));
                if (block <= 0 || widths[i] % block != 0) return false;
                const std::size_t row = ggml_row_size(static_cast<ggml_type>(types[i]), widths[i]);
                if (row > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(rows[i]))
                    return false;
                stride[i] = row * static_cast<std::size_t>(rows[i]);
                if (stride[i] > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(hm.num_experts)
                    || totals[i] <= 0 || static_cast<std::uint64_t>(totals[i])
                        != stride[i] * static_cast<std::size_t>(hm.num_experts)) return false;
            }
            return true;
        }

        bool build_graph(CacheEntry& entry, int capacity, std::size_t quota)
        {
            ggml_init_params params{};
            params.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false);
            params.no_alloc = true;
            entry.ctx = ggml_init(params);
            if (entry.ctx == nullptr) return false;
            auto* ctx = entry.ctx;
            entry.capacity = capacity;
            entry.weight[0] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(entry.type[0]),
                entry.hidden, entry.n_ff, capacity);
            entry.weight[1] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(entry.type[1]),
                entry.hidden, entry.n_ff, capacity);
            entry.weight[2] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(entry.type[2]),
                entry.n_ff, entry.hidden, capacity);
            for (auto* w : entry.weight)
            {
                // Slot bytes survive every execution of the dedicated allocator.
                ggml_set_input(w);
                ggml_set_output(w);
            }
            entry.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, entry.hidden, 1, 1);
            entry.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, entry.n_used, 1);
            entry.routes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, entry.n_used, 1);
            ggml_set_input(entry.input);
            ggml_set_input(entry.ids);
            ggml_set_input(entry.routes);

            auto* gate = ggml_mul_mat_id(ctx, entry.weight[0], entry.input, entry.ids);
            auto* up = ggml_mul_mat_id(ctx, entry.weight[1], entry.input, entry.ids);
            auto* activation = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            auto* down = ggml_mul_mat_id(ctx, entry.weight[2], activation, entry.ids);
            // Match qwen4exp's decode graph: the down output cannot alias the
            // weighted-reduction destination, making CUDA's fusion independent
            // of which freed scratch block the allocator would otherwise reuse.
            ggml_set_output(down);
            auto* weighted = ggml_mul(ctx, down, entry.routes);
            auto* sum = ggml_view_2d(ctx, weighted, entry.hidden, 1, weighted->nb[2], 0);
            for (int k = 1; k < entry.n_used; ++k)
                sum = ggml_add(ctx, sum, ggml_view_2d(ctx, weighted, entry.hidden, 1,
                    weighted->nb[2], static_cast<std::size_t>(k) * weighted->nb[1]));
            entry.output = ggml_cont(ctx, sum);
            ggml_set_output(entry.output);
            entry.graph = ggml_new_graph_custom(ctx, 256, false);
            // A disjoint UID range prevents a destroyed graph's captured CUDA
            // executable from being mistaken for a subsequently allocated graph.
            static std::atomic<std::uint64_t> next_uid{UINT64_C(0x484d4f4500000000)};
            entry.graph->uid = next_uid.fetch_add(1, std::memory_order_relaxed);
            ggml_build_forward_expand(entry.graph, entry.output);
            if (!ggml_backend_supports_op(entry.backend, gate)
                || !ggml_backend_supports_op(entry.backend, up)
                || !ggml_backend_supports_op(entry.backend, down)) return false;

            entry.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(entry.backend));
            if (entry.allocator == nullptr) return false;
            std::size_t device_bytes = 0;
            ggml_gallocr_reserve_n_size(entry.allocator, entry.graph, nullptr, nullptr, &device_bytes);
            if (device_bytes > quota || quota - device_bytes < kWorkspaceAllowance) return false;
            entry.bytes = device_bytes + kWorkspaceAllowance;
            return true;
        }

        std::unique_ptr<CacheEntry> create_entry(const HostMoeSegment& hm,
            const std::size_t (&stride)[3], ggml_backend_t backend, std::size_t quota)
        {
            std::size_t per_expert = 0;
            for (const auto s : stride)
            {
                if (s > std::numeric_limits<std::size_t>::max() - per_expert) return nullptr;
                per_expert += s;
            }
            if (quota <= kWorkspaceAllowance || per_expert == 0) return nullptr;
            std::size_t capacity = std::min<std::size_t>(hm.num_experts,
                (quota - kWorkspaceAllowance) / per_expert);
            if (capacity < static_cast<std::size_t>(hm.n_used)) return nullptr;

            // Graph scratch and quantized-tensor tail padding must fit too.
            // Shrink a measured graph before allocating any device memory.
            for (;;)
            {
                auto entry = std::make_unique<CacheEntry>();
                entry->backend = backend;
                entry->layer = hm.layer; entry->hidden = hm.hidden; entry->n_ff = hm.n_ff;
                entry->n_used = hm.n_used; entry->num_experts = hm.num_experts;
                entry->source[0] = hm.gate_data; entry->source[1] = hm.up_data; entry->source[2] = hm.down_data;
                entry->type[0] = hm.gate_type; entry->type[1] = hm.up_type; entry->type[2] = hm.down_type;
                for (int i = 0; i < 3; ++i) entry->stride[i] = stride[i];
                if (build_graph(*entry, static_cast<int>(capacity), quota)) return entry;
                if (capacity <= static_cast<std::size_t>(hm.n_used)) return nullptr;
                capacity = std::max<std::size_t>(hm.n_used, capacity * 3 / 4);
            }
        }

        bool allocate_entry(CacheEntry& entry)
        {
            std::size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(entry.backend), &free_bytes, &total_bytes);
            if (total_bytes == 0 || free_bytes <= kDeviceHeadroom
                || entry.bytes > free_bytes - kDeviceHeadroom) return false;
            entry.shared_charge = SharedCacheCharge::reserve(g_active_rank, 1, entry.bytes);
            if (!entry.shared_charge) return false;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(1)) return false;
#endif
            // Size-only reservation above creates a valid allocation plan with
            // no backing buffers. alloc_graph alone sees the matching plan and
            // does not allocate them; explicitly materialize that reservation.
            if (!ggml_gallocr_reserve(entry.allocator, entry.graph)) return false;
#if defined(TSG_GGML_TEST_HOOKS)
            entry.test_physical_bytes = ggml_gallocr_get_buffer_size(entry.allocator, 0);
            g_test_physical_bytes.fetch_add(entry.test_physical_bytes);
            if (test_failure(2)) return false;
#endif
            if (!ggml_gallocr_alloc_graph(entry.allocator, entry.graph)) return false;
            const std::size_t actual = ggml_gallocr_get_buffer_size(entry.allocator, 0);
            if (actual > entry.bytes - kWorkspaceAllowance) return false;
            entry.bytes = actual + kWorkspaceAllowance;
            // CUDA initializes each quantized tensor's over-read tail at alloc.
            // Clear its complete payload as well: unused slots must never hold
            // stale NaNs from a prior graph allocation.
            const auto errors_before = g_ggml_error_count.load(std::memory_order_acquire);
            for (auto* w : entry.weight) ggml_backend_tensor_memset(w, 0, 0, ggml_nbytes(w));
            entry.expert_for_slot.assign(entry.capacity, -1);
            entry.last_used.assign(entry.capacity, 0);
            sync_backend(entry.backend);
            if (g_ggml_error_count.load(std::memory_order_acquire) != errors_before
                || g_backend_compute_failed.load(std::memory_order_acquire)
                || !entry.shared_charge->commit(entry.bytes)) return false;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(3)) return false;
#endif
            return true;
        }

        void copy_slot(CacheEntry& entry, int slot, int expert)
        {
            std::vector<std::pair<std::size_t, std::size_t>> pieces;
            for (int i = 0; i < 3; ++i)
            {
                const auto* src = static_cast<const std::uint8_t*>(entry.source[i])
                    + static_cast<std::size_t>(expert) * entry.stride[i];
                // Respect prior registrations from streamed prefill without
                // pinning the whole expert stack or creating a second RAM copy.
                host_pin_split(src, entry.stride[i], pieces);
                for (const auto& p : pieces)
                    ggml_backend_tensor_set_async(entry.backend, entry.weight[i], src + p.first,
                        static_cast<std::size_t>(slot) * entry.stride[i] + p.first, p.second);
            }
        }

        ggml_backend_buffer_t tensor_buffer(const ggml_tensor* tensor)
        {
            while (tensor != nullptr && tensor->buffer == nullptr) tensor = tensor->view_src;
            return tensor != nullptr ? tensor->buffer : nullptr;
        }

        bool device_source(const ggml_tensor* tensor, std::size_t bytes, ggml_backend_t backend)
        {
            const auto buffer = tensor_buffer(tensor);
            return tensor != nullptr && tensor->type == GGML_TYPE_F32 && tensor->data != nullptr
                && ggml_is_contiguous(tensor) && ggml_nbytes(tensor) >= bytes
                && buffer != nullptr && ggml_backend_buffer_get_type(buffer)
                    == ggml_backend_get_default_buffer_type(backend);
        }

        void copy_device_row(ggml_backend_t backend, const ggml_tensor* source,
            ggml_tensor* destination, std::size_t offset, bool destination_row = false)
        {
            // A local copy descriptor has the destination's scalar layout and
            // points at one contiguous source row. It never changes tensors in
            // the outer graph, and the public backend copy queues on the same
            // CUDA stream as expert uploads and graph execution.
            ggml_tensor row = destination_row ? *source : *destination;
            const auto* storage = destination_row ? destination : source;
            row.buffer = tensor_buffer(storage);
            row.data = static_cast<std::uint8_t*>(storage->data) + offset;
            row.view_src = nullptr;
            row.view_offs = 0;
            row.extra = nullptr;
            if (destination_row) ggml_backend_tensor_copy_async(backend, backend, source, &row);
            else ggml_backend_tensor_copy_async(backend, backend, &row, destination);
        }
#endif
    }

    int host_moe_cached_experts(const HostMoeSegment& hm, const float* x,
        const std::int32_t* ids, const float* routes, float* out, const char* kernel_name)
    {
#ifdef TSG_GGML_USE_CUDA
        const std::size_t budget = cache_budget(), layers = cache_layer_count();
        const bool device_inputs = x == nullptr && routes == nullptr;
        const bool device_output = out == nullptr;
        if (budget == 0 || layers == 0 || kernel_name == nullptr
            || std::strcmp(kernel_name, "qwen4exp fused token span") != 0
            || g_backend == nullptr || !ggml_backend_is_cuda(g_backend)
            || (!device_inputs && (x == nullptr || routes == nullptr))
            || ids == nullptr || (device_output && !device_inputs)) return 0;
        std::size_t stride[3] = {};
        if (!valid_layout(hm, stride)) return 0;
        // Validate the complete short block before allocating or computing any
        // row. Unsupported data always falls back as one block; execution
        // failure returns an error and never mixes CPU/GPU expert arithmetic.
        const std::size_t route_count = static_cast<std::size_t>(hm.n_used) * hm.seq_len;
        if (device_inputs && (!device_source(hm.moe_in,
                static_cast<std::size_t>(hm.hidden) * hm.seq_len * sizeof(float), g_backend)
            || !device_source(hm.weights, route_count * sizeof(float), g_backend))) return 0;
        if (device_output && !device_source(hm.moe_out,
                static_cast<std::size_t>(hm.hidden) * hm.seq_len * sizeof(float), g_backend)) return 0;
        for (std::size_t k = 0; k < route_count; ++k)
            if (ids[k] < 0 || ids[k] >= hm.num_experts
                || (!device_inputs && !std::isfinite(routes[k]))) return 0;

        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto found = g_entries.begin();
        while (found != g_entries.end() && !(*found)->matches(hm, g_backend)) ++found;
        if (found == g_entries.end())
        {
            auto entry = create_entry(hm, stride, g_backend, budget / layers);
            if (!entry) return 0;
            while (g_reserved > budget - entry->bytes && !g_entries.empty())
            {
                const auto retired_bytes = g_entries.back()->bytes;
                g_entries.pop_back();
                g_reserved -= retired_bytes;
            }
            if (g_reserved > budget - entry->bytes || !allocate_entry(*entry)) return 0;
            const auto bytes = entry->bytes;
            // A list-node allocation can throw. Publish before changing private
            // accounting so unique_ptr rollback still owns every physical byte.
            g_entries.push_front(std::move(entry));
            g_reserved += bytes;
            const auto& published = *g_entries.front();
            if (diagnostics()) std::fprintf(stderr,
                "[HOSTMOE-CACHE] layer=%d slots=%d reserved=%zu budget=%zu graph=%zu workspace_allowance=%zu input_bridge=%s\n",
                hm.layer, published.capacity, g_reserved, budget,
                published.bytes - kWorkspaceAllowance, kWorkspaceAllowance, device_inputs ? "device" : "host");
        }
        else
        {
            g_entries.splice(g_entries.begin(), g_entries, found);
        }
        auto& entry = *g_entries.front();
        std::vector<int> remapped(hm.n_used, -1);
        std::vector<std::uint8_t> protected_slots(entry.capacity, 0);
        std::vector<std::pair<int, int>> misses;
        misses.reserve(hm.n_used);
        static const bool prefetch = [] {
            const char* e = std::getenv("TS_HOST_MOE_EXPERT_CACHE_PREFETCH");
            return e != nullptr && e[0] == '1';
        }();
        // Qwen4Exp's precision policy uses decode kernels for every row in a
        // short prefill/speculative target block. Replay this same scalar graph
        // so compact-cache verification has exactly the decode rounding. The
        // one allocation above serves every row; eviction only replaces slots.
        for (int row = 0; row < hm.seq_len; ++row)
        {
            const auto* row_ids = ids + static_cast<std::size_t>(row) * hm.n_used;
            std::fill(remapped.begin(), remapped.end(), -1);
            std::fill(protected_slots.begin(), protected_slots.end(), 0);
            misses.clear();
            // Protect ALL hits before filling any miss, including hits later in
            // the router's order. Otherwise an early miss can evict a later hit.
            for (int k = 0; k < hm.n_used; ++k)
                for (int s = 0; s < entry.capacity; ++s)
                    if (entry.expert_for_slot[s] == row_ids[k])
                    {
                        remapped[k] = s;
                        protected_slots[s] = 1;
                        ++g_hits;
                        break;
                    }
            for (int k = 0; k < hm.n_used; ++k)
            {
                if (remapped[k] >= 0) continue;
                // Duplicated selected IDs are legal for the helper, even though
                // Qwen's top-k normally produces distinct experts.
                for (int previous = 0; previous < k; ++previous)
                    if (row_ids[previous] == row_ids[k]) { remapped[k] = remapped[previous]; break; }
                if (remapped[k] >= 0) { ++g_hits; continue; }
                int slot = -1;
                for (int s = 0; s < entry.capacity; ++s)
                    if (!protected_slots[s] && (slot < 0 || entry.last_used[s] < entry.last_used[slot])) slot = s;
                if (slot < 0)
                {
                    set_last_error("qwen4exp expert cache: all slots protected before routed experts were filled.");
                    return -1;
                }
                misses.emplace_back(slot, row_ids[k]);
                remapped[k] = slot;
                protected_slots[slot] = 1;
                ++g_misses;
            }
            // Fault all selected misses concurrently before pageable CUDA
            // uploads serialize source page faults on the launching thread.
            // Only raw bytes are touched; cache-hit payloads and row arithmetic
            // are unchanged. Keep this opt-in for cold/warm trained A/B runs.
            if (prefetch && !misses.empty())
            {
                std::vector<std::pair<std::size_t, std::size_t>> ranges;
                ranges.reserve(misses.size());
                for (int part = 0; part < 3; ++part)
                {
                    ranges.clear();
                    for (const auto& miss : misses)
                        ranges.emplace_back(static_cast<std::size_t>(miss.second) * entry.stride[part],
                            entry.stride[part]);
                    std::sort(ranges.begin(), ranges.end());
                    prefetch_mapped_ranges(static_cast<const std::uint8_t*>(entry.source[part]), ranges);
                }
            }
            for (const auto& miss : misses)
            {
                copy_slot(entry, miss.first, miss.second);
                entry.expert_for_slot[miss.first] = miss.second;
            }
            for (const int slot : remapped) entry.last_used[slot] = ++entry.clock;
            // Copies and compute use the backend's own stream; graph topology and
            // addresses never change when slots are replaced or IDs are remapped.
            if (device_inputs)
            {
                copy_device_row(entry.backend, hm.moe_in, entry.input,
                    static_cast<std::size_t>(row) * entry.hidden * sizeof(float));
                copy_device_row(entry.backend, hm.weights, entry.routes,
                    static_cast<std::size_t>(row) * entry.n_used * sizeof(float));
            }
            else
            {
                ggml_backend_tensor_set_async(entry.backend, entry.input,
                    x + static_cast<std::size_t>(row) * hm.hidden, 0,
                    static_cast<std::size_t>(entry.hidden) * sizeof(float));
                ggml_backend_tensor_set_async(entry.backend, entry.routes,
                    routes + static_cast<std::size_t>(row) * hm.n_used, 0,
                    static_cast<std::size_t>(entry.n_used) * sizeof(float));
            }
            ggml_backend_tensor_set_async(entry.backend, entry.ids, remapped.data(), 0,
                static_cast<std::size_t>(entry.n_used) * sizeof(std::int32_t));
            if (compute_graph(entry.backend, entry.graph) != GGML_STATUS_SUCCESS
                || g_backend_compute_failed.load(std::memory_order_acquire))
            {
                set_last_error("qwen4exp expert cache: CUDA expert graph failed.");
                return -1;
            }
            if (device_output)
                copy_device_row(entry.backend, entry.output, hm.moe_out,
                    static_cast<std::size_t>(row) * entry.hidden * sizeof(float), true);
            else
                ggml_backend_tensor_get(entry.output, out + static_cast<std::size_t>(row) * hm.hidden,
                    0, static_cast<std::size_t>(entry.hidden) * sizeof(float));
            ++g_calls;
        }
        return 1;
#else
        (void)hm; (void)x; (void)ids; (void)routes; (void)out; (void)kernel_name;
        return 0;
#endif
    }

    void host_moe_expert_cache_on_drop(const void* ptr)
    {
        if (ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (auto it = g_entries.begin(); it != g_entries.end();)
        {
            const auto& entry = **it;
            if (entry.source[0] == ptr || entry.source[1] == ptr || entry.source[2] == ptr)
            {
                const auto retired_bytes = entry.bytes;
                it = g_entries.erase(it);
                g_reserved -= retired_bytes;
            }
            else ++it;
        }
    }

    void host_moe_expert_cache_release()
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (diagnostics() && (g_calls != 0 || g_reserved != 0)) std::fprintf(stderr,
            "[HOSTMOE-CACHE] calls=%llu hits=%llu misses=%llu reserved=%zu budget=%zu release\n",
            static_cast<unsigned long long>(g_calls), static_cast<unsigned long long>(g_hits),
            static_cast<unsigned long long>(g_misses), g_reserved, cache_budget());
        g_entries.clear();
        g_reserved = 0;
        g_hits = g_misses = g_calls = 0;
    }

    void host_moe_expert_cache_stats(std::int64_t* reserved, std::int64_t* budget,
        std::int64_t* hits, std::int64_t* misses, std::int64_t* calls)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        *reserved = static_cast<std::int64_t>(g_reserved);
        *budget = static_cast<std::int64_t>(cache_budget());
        *hits = static_cast<std::int64_t>(g_hits);
        *misses = static_cast<std::int64_t>(g_misses);
        *calls = static_cast<std::int64_t>(g_calls);
    }

#if defined(TSG_GGML_TEST_HOOKS)
    // Linked only into the standalone cache fixture, never a production export.
    void host_moe_expert_cache_test_fail_next(int stage) { g_test_failure_stage.store(stage); }
    std::size_t host_moe_expert_cache_test_physical_bytes() { return g_test_physical_bytes.load(); }
#endif
}

TSG_EXPORT int TSGgml_HostMoeExpertCacheStats(std::int64_t* reserved, std::int64_t* budget,
    std::int64_t* hits, std::int64_t* misses, std::int64_t* calls)
{
    if (!reserved || !budget || !hits || !misses || !calls) return 0;
    tsg::host_moe_expert_cache_stats(reserved, budget, hits, misses, calls);
    return 1;
}
