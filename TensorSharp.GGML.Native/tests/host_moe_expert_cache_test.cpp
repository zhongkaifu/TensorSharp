// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// The optional expert tier against an independent full-resident CUDA graph.
#include "ggml_ops_internal.h"
#include "ggml_ops_precision_policy.h"
#include "ggml_ops_shared_cache_budget.h"
#include "ggml-impl.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>

extern "C" int TSGgml_HostMoeExpertCacheStats(std::int64_t*, std::int64_t*,
    std::int64_t*, std::int64_t*, std::int64_t*);

// This standalone target compiles the production cache implementation with
// small bridge stubs; it never loads a model or depends on modified ggml code.
namespace tsg
{
    DeviceState g_device_states[TSG_MAX_DEVICES];
    thread_local int g_active_rank = 0;
    std::atomic<std::uint64_t> g_ggml_error_count{0};
    std::atomic<bool> g_backend_compute_failed{false};
    std::uint64_t g_test_prefetch_calls = 0;
    void set_last_error(const std::string& message) { std::fprintf(stderr, "%s\n", message.c_str()); }
    // Expert tensors in this fixture have no registered dense-Qwen precision
    // policy. Keep their ordinary unchanged ggml CUDA dispatch.
    ggml_backend_t q8_f32_execution_backend(ggml_backend_t backend, ggml_cgraph*) { return backend; }
    void host_moe_expert_cache_test_fail_next(int stage);
    std::size_t host_moe_expert_cache_test_physical_bytes();
#ifdef TSG_GGML_USE_CUDA
    void host_pin_split(const void*, std::size_t bytes,
        std::vector<std::pair<std::size_t, std::size_t>>& pieces)
    {
        pieces.assign(1, {0, bytes});
    }
    void prefetch_mapped_ranges(const std::uint8_t* base,
        const std::vector<std::pair<std::size_t, std::size_t>>& ranges)
    {
        // The production bridge reuses the existing MoE page-fault pool. This
        // standalone cache test observes its raw ranges and touches them without
        // linking the unrelated host-FFN engine or needing mapped model files.
        if (base == nullptr || ranges.empty() || !std::is_sorted(ranges.begin(), ranges.end()))
            throw std::runtime_error("Cache prefetch ranges are missing or unordered");
        volatile std::uint8_t sink = 0;
        for (const auto& range : ranges)
        {
            if (range.second == 0) throw std::runtime_error("Cache prefetch range is empty");
            sink ^= base[range.first];
            sink ^= base[range.first + range.second - 1];
        }
        (void)sink;
        ++g_test_prefetch_calls;
    }
#endif
}

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition) throw std::runtime_error(message);
    }

    void set_environment(const char* key, const char* value)
    {
#if defined(_WIN32)
        _putenv_s(key, value);
#else
        setenv(key, value, 1);
#endif
    }

    struct Stats
    {
        std::int64_t reserved = 0, budget = 0, hits = 0, misses = 0, calls = 0;
        Stats()
        {
            require(TSGgml_HostMoeExpertCacheStats(&reserved, &budget, &hits, &misses, &calls) == 1,
                "Could not read expert-cache engagement counters");
            require(reserved >= 0 && budget >= 0 && reserved <= budget,
                "Expert-cache allocations exceeded the explicit byte ceiling");
        }
    };

    constexpr int hidden = 256, ff = 256, experts = 16, used = 2;

    struct SharedLedger
    {
        struct Allocation { std::int64_t bytes; bool committed = false; };
        std::int64_t capacity = 2 << 20, pending = 0, committed = 0;
        std::uint64_t next = 0, reserve_calls = 0, commit_calls = 0, release_calls = 0;
        bool fail_commit = false, valid = true, attached = false;
        std::map<std::uint64_t, Allocation> allocations;

        ~SharedLedger()
        {
            // Keep callback context alive during exception unwinding too.
            if (attached)
            {
                tsg::host_moe_expert_cache_release();
                tsg::SharedCacheCharge::detach(this);
            }
        }

        static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes)
        {
            auto& l = *static_cast<SharedLedger*>(context);
            ++l.reserve_calls;
            if (rank != 0 || kind != 1 || bytes <= 0) { l.valid = false; return 0; }
            if (bytes > l.capacity - l.pending - l.committed) return 0;
            const auto token = ++l.next;
            l.allocations.emplace(token, Allocation{bytes});
            l.pending += bytes;
            return token;
        }
        static int commit(void* context, std::uint64_t token)
        {
            auto& l = *static_cast<SharedLedger*>(context);
            ++l.commit_calls;
            auto found = l.allocations.find(token);
            if (found == l.allocations.end() || found->second.committed
                || tsg::host_moe_expert_cache_test_physical_bytes() == 0) { l.valid = false; return 0; }
            if (l.fail_commit) return 0;
            auto& allocation = found->second;
            l.pending -= allocation.bytes;
            l.committed += allocation.bytes;
            allocation.committed = true;
            return 1;
        }
        static void release(void* context, std::uint64_t token)
        {
            auto& l = *static_cast<SharedLedger*>(context);
            ++l.release_calls;
            auto found = l.allocations.find(token);
            // These ownership fixtures keep one cache entry alive at a time;
            // the separate full-resident oracle is not charged to this cache.
            if (found == l.allocations.end() || tsg::host_moe_expert_cache_test_physical_bytes() != 0)
            { l.valid = false; return; }
            auto& allocation = found->second;
            (allocation.committed ? l.committed : l.pending) -= allocation.bytes;
            l.allocations.erase(found);
        }
        bool attach()
        {
            const bool result = tsg::SharedCacheCharge::attach(this, reserve, commit, release);
            if (result) attached = true;
            return result;
        }
        bool detach()
        {
            if (!tsg::SharedCacheCharge::detach(this)) return false;
            attached = false;
            return true;
        }
        void check_empty()
        {
            require(valid && allocations.empty() && pending == 0 && committed == 0,
                "Expert-cache shared credit escaped physical ownership or rollback");
            require(tsg::host_moe_expert_cache_test_physical_bytes() == 0 && Stats().reserved == 0,
                "Expert-cache rollback retained physical or private-accounted storage");
        }
    };

    struct Weights
    {
        tsg::HostMoeSegment segment;
        std::array<std::vector<std::uint8_t>, 3> quantized;
        std::array<ggml_type, 3> types{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};

        void fill(float seed)
        {
            std::vector<float> values(static_cast<std::size_t>(hidden) * ff * experts);
            std::vector<float> importance(hidden, 1.0f);
            for (int part = 0; part < 3; ++part)
            {
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = 0.04f * std::sin(seed + float(i % 7919) * 0.013f + part * 0.7f);
                const auto bytes = ggml_row_size(types[part], hidden) * ff * experts;
                quantized[part].resize(bytes);
                const auto written = ggml_quantize_chunk(types[part], values.data(),
                    quantized[part].data(), 0, ff * experts, hidden,
                    ggml_quantize_requires_imatrix(types[part]) ? importance.data() : nullptr);
                require(written == bytes, "Synthetic quantized expert layout is incomplete");
            }
            segment.layer = 0;
            segment.hidden = hidden; segment.n_ff = ff; segment.num_experts = experts;
            segment.n_used = used; segment.seq_len = 1; segment.activation = 0;
            segment.gate_data = quantized[0].data(); segment.gate_type = types[0];
            segment.gate_ne0 = hidden; segment.gate_ne1 = ff; segment.gate_bytes = quantized[0].size();
            segment.up_data = quantized[1].data(); segment.up_type = types[1];
            segment.up_ne0 = hidden; segment.up_ne1 = ff; segment.up_bytes = quantized[1].size();
            segment.down_data = quantized[2].data(); segment.down_type = types[2];
            segment.down_ne0 = ff; segment.down_ne1 = hidden; segment.down_bytes = quantized[2].size();
        }
    };

    void fallback_checks(Weights& weights)
    {
        std::array<float, hidden> x{}, result{};
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.6f, 0.4f};
        auto hm = weights.segment;
        const auto run = [&] { return tsg::host_moe_cached_experts(hm, x.data(), ids.data(),
            routes.data(), result.data(), "qwen4exp fused token span"); };
        hm.seq_len = static_cast<int>(TSG_PRECISION_DECODE_COLUMNS) + 1;
        require(run() == 0, "Long prefill must retain the existing offload path");
        hm.seq_len = 1; hm.activation = 1;
        require(run() == 0, "Non-SiLU activation must retain the existing offload path");
        hm.activation = 0; hm.up_bias = routes.data();
        require(run() == 0, "Biased experts must retain the existing offload path");
        hm.up_bias = nullptr;
        require(tsg::host_moe_cached_experts(hm, x.data(), ids.data(), routes.data(), result.data(),
            "Other model") == 0, "The opt-in cache must not change other model families");
    }

#ifdef TSG_GGML_USE_CUDA
    struct DeviceInputs
    {
        ggml_context* context = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* x = nullptr;
        ggml_tensor* routes = nullptr;
        ggml_tensor* output = nullptr;

        DeviceInputs()
        {
            context = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
            require(context != nullptr, "Could not allocate device-bridge input metadata");
            x = ggml_new_tensor_2d(context, GGML_TYPE_F32, hidden, TSG_PRECISION_DECODE_COLUMNS);
            routes = ggml_new_tensor_2d(context, GGML_TYPE_F32, used, TSG_PRECISION_DECODE_COLUMNS);
            output = ggml_new_tensor_2d(context, GGML_TYPE_F32, hidden, TSG_PRECISION_DECODE_COLUMNS);
            buffer = ggml_backend_alloc_ctx_tensors(context, g_backend);
            require(buffer != nullptr, "Could not allocate device-bridge inputs");
        }

        ~DeviceInputs()
        {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (context) ggml_free(context);
        }
    };

    struct FullResidentGraph
    {
        ggml_context* context = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* x = nullptr;
        ggml_tensor* ids = nullptr;
        ggml_tensor* routes = nullptr;
        ggml_tensor* output = nullptr;

        explicit FullResidentGraph(const Weights& weights)
        {
            context = ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true});
            require(context != nullptr, "Could not allocate reference tensor metadata");
            auto* gate = ggml_new_tensor_3d(context, weights.types[0], hidden, ff, experts);
            auto* up = ggml_new_tensor_3d(context, weights.types[1], hidden, ff, experts);
            auto* down = ggml_new_tensor_3d(context, weights.types[2], ff, hidden, experts);
            for (auto* t : {gate, up, down}) { ggml_set_input(t); ggml_set_output(t); }
            x = ggml_new_tensor_3d(context, GGML_TYPE_F32, hidden, 1, 1);
            ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, used, 1);
            routes = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, used, 1);
            for (auto* t : {x, ids, routes}) ggml_set_input(t);
            // The outer Qwen host-MoE boundary keeps these source tensors live
            // as graph outputs. Mirror that lifetime when testing D2D reads
            // after the full-resident reference graph has finished computing.
            ggml_set_output(x);
            ggml_set_output(routes);
            auto* g = ggml_mul_mat_id(context, gate, x, ids);
            auto* u = ggml_mul_mat_id(context, up, x, ids);
            auto* h = ggml_mul(context, ggml_silu(context, g), u);
            auto* parts = ggml_mul_mat_id(context, down, h, ids);
            ggml_set_output(parts);
            auto* weighted = ggml_mul(context, parts, routes);
            auto* sum = ggml_view_2d(context, weighted, hidden, 1, weighted->nb[2], 0);
            for (int k = 1; k < used; ++k)
                sum = ggml_add(context, sum, ggml_view_2d(context, weighted, hidden, 1,
                    weighted->nb[2], static_cast<std::size_t>(k) * weighted->nb[1]));
            output = ggml_cont(context, sum);
            ggml_set_output(output);
            graph = ggml_new_graph_custom(context, 64, false);
            ggml_build_forward_expand(graph, output);
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
            require(allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph),
                "Could not allocate full-resident reference graph");
            ggml_backend_tensor_set(gate, weights.quantized[0].data(), 0, weights.quantized[0].size());
            ggml_backend_tensor_set(up, weights.quantized[1].data(), 0, weights.quantized[1].size());
            ggml_backend_tensor_set(down, weights.quantized[2].data(), 0, weights.quantized[2].size());
        }

        ~FullResidentGraph()
        {
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
        }

        std::array<float, hidden> compute(const std::array<float, hidden>& activation,
            const std::array<std::int32_t, used>& selected, const std::array<float, used>& probabilities)
        {
            ggml_backend_tensor_set(x, activation.data(), 0, sizeof(activation));
            ggml_backend_tensor_set(ids, selected.data(), 0, sizeof(selected));
            ggml_backend_tensor_set(routes, probabilities.data(), 0, sizeof(probabilities));
            require(ggml_backend_graph_compute(g_backend, graph) == GGML_STATUS_SUCCESS,
                "Full-resident reference expert graph failed");
            std::array<float, hidden> result{};
            ggml_backend_tensor_get(output, result.data(), 0, sizeof(result));
            return result;
        }
    };

    void shared_budget_checks(Weights& weights)
    {
        using tsg::SharedCacheCharge;
        SharedLedger ledger;
        std::array<float, hidden> x{}, output{};
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.6f, 0.4f};
        auto call = [&] {
            return tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(),
                output.data(), "qwen4exp fused token span");
        };
        require(call() == 1 && Stats().reserved > 0, "Unconfigured expert cache did not engage");
        require(!ledger.attach(), "Budget attach adopted an already-live uncharged expert graph");
        tsg::host_moe_expert_cache_release();
        require(ledger.attach(), "Could not attach budget after expert-cache physical teardown");

        // Quota refusal must precede even the first native-allocation hook.
        ledger.capacity = 0;
        output.fill(-999.0f);
        tsg::host_moe_expert_cache_test_fail_next(1);
        require(call() == 0, "Shared quota refusal did not retain the normal fallback");
        ledger.check_empty();
        require(std::all_of(output.begin(), output.end(), [](float value) { return value == -999.0f; }),
            "Refused expert-cache creation overwrote caller output");
        ledger.capacity = 2 << 20;
        require(call() == 0, "Denied quota consumed the later allocation-failure hook");
        ledger.check_empty();

        for (int stage : {1, 2, 3})
        {
            tsg::host_moe_expert_cache_test_fail_next(stage);
            require(call() == 0, "Injected expert-cache allocation/publication failure was ignored");
            ledger.check_empty();
        }
        ledger.fail_commit = true;
        require(call() == 0, "Refused shared commit published an expert-cache entry");
        ledger.check_empty();
        ledger.fail_commit = false;

        require(call() == 1, "Budgeted expert cache did not recover after allocation rollback");
        const auto initial = Stats();
        require(ledger.valid && ledger.pending == 0 && ledger.committed >= initial.reserved,
            "Committed expert graph is not covered by shared device credit");
        require(!ledger.detach(), "Detached callbacks with a live expert graph");
        const auto reservations = ledger.reserve_calls;
        const auto commits = ledger.commit_calls;
        require(call() == 1 && ledger.reserve_calls == reservations && ledger.commit_calls == commits,
            "Warm expert reuse acquired duplicate shared reservations");
        const auto exact_bytes = ledger.committed;
        tsg::host_moe_expert_cache_on_drop(weights.segment.gate_data);
        ledger.check_empty();
        ledger.capacity = exact_bytes - 1;
        require(call() == 0, "Expert-cache graph exceeded shared capacity by one byte");
        ledger.check_empty();
        ledger.capacity = exact_bytes;
        require(call() == 1, "Budgeted expert graph failed to reload after invalidation");
        tsg::host_moe_expert_cache_release();
        ledger.check_empty();
        require(ledger.detach(), "Cannot detach after all expert-cache buffers were freed");
        std::puts("PASS: expert shared quota, reserve-before-allocation, rollback, commit, reuse, invalidation and physical-free order");
    }

    void cuda_checks(Weights& weights)
    {
        fallback_checks(weights);
        FullResidentGraph reference(weights);
        std::array<float, hidden> x{}, actual{};
        std::array<float, used> routes{0.65f, 0.35f};
        // A 2 MiB quota holds fewer than 16 experts (1 MiB is the workspace
        // reserve). Cycling sixteen IDs therefore requires real slot eviction.
        for (int iteration = 0; iteration < 80; ++iteration)
        {
            for (int i = 0; i < hidden; ++i)
                x[i] = std::sin(0.03f * i + 0.1f * iteration);
            std::array<std::int32_t, used> ids{(iteration * 3) % experts, (iteration * 3 + 1) % experts};
            if (iteration % 5 == 0) ids = {0, 2}; // miss can precede a later protected hit
            const auto expected = reference.compute(x, ids, routes);
            require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(),
                actual.data(), "qwen4exp fused token span") == 1,
                "The opt-in CUDA expert cache did not engage");
            require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
                "Compact/remapped or evicted expert results differ from the full-resident CUDA graph");
            const auto again = reference.compute(x, ids, routes);
            require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(),
                actual.data(), "qwen4exp fused token span") == 1,
                "Warm expert slots did not engage");
            require(std::memcmp(actual.data(), again.data(), sizeof(actual)) == 0,
                "Warm expert slots reused stale activations or different arithmetic");
            auto device_hm = weights.segment;
            device_hm.moe_in = reference.x;
            device_hm.weights = reference.routes;
            require(tsg::host_moe_cached_experts(device_hm, nullptr, ids.data(), nullptr,
                actual.data(), "qwen4exp fused token span") == 1,
                "The CUDA device-to-device scalar input bridge did not engage");
            require(std::memcmp(actual.data(), again.data(), sizeof(actual)) == 0,
                "CUDA device-to-device scalar inputs changed expert arithmetic");
            device_hm.moe_out = reference.output;
            require(tsg::host_moe_cached_experts(device_hm, nullptr, ids.data(), nullptr,
                nullptr, "qwen4exp fused token span") == 1,
                "The CUDA device-to-device scalar output bridge did not engage");
            tsg::sync_backend(g_backend);
            ggml_backend_tensor_get(reference.output, actual.data(), 0, sizeof(actual));
            require(std::memcmp(actual.data(), again.data(), sizeof(actual)) == 0,
                "CUDA device-to-device scalar output changed expert arithmetic");
            Stats stats;
            require(stats.reserved > 0, "Engaged cache did not account for its device buffers");
        }
        const Stats warm;
        require(warm.calls == 320 && warm.hits >= 480 && warm.misses > experts,
            "Test did not exercise warm hits and actual expert eviction");

        // Short prefill and speculative target verification must execute the
        // same one-row CUDA FFN arithmetic as successive autoregressive calls.
        // Changing activations, IDs, routes and repeated IDs exercises reuse
        // and slot eviction between rows inside a single cache-helper call.
        DeviceInputs device_inputs;
        for (int rows = 2; rows <= TSG_PRECISION_DECODE_COLUMNS; ++rows)
        {
            auto hm = weights.segment;
            hm.seq_len = rows;
            std::vector<float> inputs(static_cast<std::size_t>(rows) * hidden);
            std::vector<std::int32_t> selected(static_cast<std::size_t>(rows) * used);
            std::vector<float> probabilities(static_cast<std::size_t>(rows) * used);
            std::vector<float> expected(static_cast<std::size_t>(rows) * hidden);
            std::vector<float> block_result(expected.size(), -999.0f);
            for (int row = 0; row < rows; ++row)
            {
                for (int i = 0; i < hidden; ++i)
                    x[i] = inputs[static_cast<std::size_t>(row) * hidden + i]
                        = std::sin(0.07f * i + rows * 0.2f + row);
                std::array<std::int32_t, used> row_ids{(row * 5 + rows) % experts,
                    (row * 5 + rows + 1) % experts};
                if (row % 3 == 0) row_ids[1] = row_ids[0];
                std::array<float, used> row_routes{0.25f + 0.05f * row, 0.75f - 0.05f * row};
                std::copy(row_ids.begin(), row_ids.end(), selected.begin() + row * used);
                std::copy(row_routes.begin(), row_routes.end(), probabilities.begin() + row * used);
                const auto row_result = reference.compute(x, row_ids, row_routes);
                std::copy(row_result.begin(), row_result.end(), expected.begin() + row * hidden);
            }
            const auto before = Stats();
            require(tsg::host_moe_cached_experts(hm, inputs.data(), selected.data(), probabilities.data(),
                block_result.data(), "qwen4exp fused token span") == 1,
                "Short-block expert cache did not replay the scalar CUDA graph");
            require(std::memcmp(block_result.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "Short-block expert rows differ from full-resident decode rounding");
            require(Stats().calls == before.calls + rows,
                "Short-block cache did not compute every row through the scalar graph");

            ggml_backend_tensor_set(device_inputs.x, inputs.data(), 0, inputs.size() * sizeof(float));
            ggml_backend_tensor_set(device_inputs.routes, probabilities.data(), 0, probabilities.size() * sizeof(float));
            hm.moe_in = device_inputs.x;
            hm.weights = device_inputs.routes;
            std::fill(block_result.begin(), block_result.end(), -999.0f);
            const auto device_before = Stats();
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                block_result.data(), "qwen4exp fused token span") == 1,
                "Short-block CUDA device-to-device input bridge did not engage");
            require(std::memcmp(block_result.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "Short-block CUDA device-to-device rows changed expert arithmetic");
            require(Stats().calls == device_before.calls + rows,
                "Short-block device bridge did not compute every row");
            hm.moe_out = device_inputs.output;
            std::fill(block_result.begin(), block_result.end(), -999.0f);
            const auto output_before = Stats();
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                nullptr, "qwen4exp fused token span") == 1,
                "Short-block CUDA device-to-device output bridge did not engage");
            // Production consumes this queued copy in the next graph on the
            // same backend stream. A standalone host read uses its own CUDA
            // stream and must wait for that pending copy explicitly.
            tsg::sync_backend(g_backend);
            ggml_backend_tensor_get(device_inputs.output, block_result.data(), 0,
                block_result.size() * sizeof(float));
            require(std::memcmp(block_result.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "Short-block CUDA device-to-device output changed expert arithmetic");
            require(Stats().calls == output_before.calls + rows,
                "Short-block device output did not compute every row");

            // A bad later row must reject the complete block before computing
            // the valid early rows or overwriting any caller output.
            selected.back() = experts;
            std::fill(block_result.begin(), block_result.end(), -999.0f);
            const auto invalid_before = Stats();
            require(tsg::host_moe_cached_experts(hm, inputs.data(), selected.data(), probabilities.data(),
                block_result.data(), "qwen4exp fused token span") == 0,
                "Invalid later-row expert ID did not reject the entire block");
            require(std::all_of(block_result.begin(), block_result.end(), [](float v) { return v == -999.0f; }),
                "Rejected short block exposed a partially computed CUDA result");
            const auto invalid_after = Stats();
            require(invalid_after.calls == invalid_before.calls && invalid_after.hits == invalid_before.hits
                && invalid_after.misses == invalid_before.misses && invalid_after.reserved == invalid_before.reserved,
                "Rejected short block mutated cache state before validating every row");
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                block_result.data(), "qwen4exp fused token span") == 0,
                "Device bridge did not reject an invalid later-row expert ID");
            require(std::all_of(block_result.begin(), block_result.end(), [](float v) { return v == -999.0f; }),
                "Rejected device-bridge block exposed a partially computed CUDA result");

            selected.back() = 0;
            hm.weights = nullptr;
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                block_result.data(), "qwen4exp fused token span") == 0,
                "Missing device routing source did not retain the staged fallback");
        }

        tsg::host_moe_expert_cache_on_drop(weights.segment.gate_data);
        require(Stats().reserved == 0, "Invalidated model weights retained compact CUDA slots");
        // Rewrite the same vector allocations, as a reload can recycle an mmap
        // address. Invalidation must force fresh copies of every routed expert.
        weights.fill(1.4f);
        FullResidentGraph reloaded(weights);
        std::array<std::int32_t, used> ids{0, 2};
        const auto expected = reloaded.compute(x, ids, routes);
        require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(), actual.data(),
            "qwen4exp fused token span") == 1, "Reloaded cache did not engage");
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "Reloaded expert cache read bytes from the previous model identity");
        auto output_hm = weights.segment;
        output_hm.moe_in = reloaded.x;
        output_hm.weights = reloaded.routes;
        output_hm.moe_out = reloaded.output;
        require(tsg::host_moe_cached_experts(output_hm, nullptr, ids.data(), nullptr,
            nullptr, "qwen4exp fused token span") == 1, "Reloaded direct output did not engage");
        // No tensor read/synchronize between the final queued row copy and
        // invalidation. Entry retirement must drain that copy before freeing
        // its source graph buffer, leaving the external destination valid.
        tsg::host_moe_expert_cache_on_drop(weights.segment.gate_data);
        require(Stats().reserved == 0, "Direct output prevented cache source invalidation");
        ggml_backend_tensor_get(reloaded.output, actual.data(), 0, sizeof(actual));
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "Cache invalidation freed a pending direct-output copy's source");
        tsg::host_moe_expert_cache_release();
        const Stats released;
        require(released.reserved == 0 && released.calls == 0 && released.hits == 0 && released.misses == 0,
            "Explicit expert-cache teardown retained device allocations or old counters");
    }
#endif
}

int main(int argc, char** argv)
{
    try
    {
        const bool invalid_budget = argc == 3 && std::strcmp(argv[1], "--invalid-budget") == 0;
        set_environment("TS_HOST_MOE_EXPERT_CACHE_MB", invalid_budget ? argv[2] : "2");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_LAYERS", "1");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS", "1");
        if (invalid_budget)
        {
            const auto stats = Stats();
            require(stats.budget == 0 && stats.reserved == 0 && stats.calls == 0,
                "Malformed expert-cache budget was accepted by the native parser");
            std::puts("PASS: malformed native cache budget remains disabled");
            return 0;
        }
        Weights weights;
        bool cpu_only = false, prefetch = false;
        for (int arg = 1; arg < argc; ++arg)
        {
            if (std::strcmp(argv[arg], "--cpu") == 0) cpu_only = true;
            else if (std::strcmp(argv[arg], "--prefetch") == 0) prefetch = true;
            else if (std::strcmp(argv[arg], "--quantization") == 0 && arg + 1 < argc)
            {
                ggml_type type = GGML_TYPE_COUNT;
                ++arg;
                if (std::strcmp(argv[arg], "iq1_m") == 0) type = GGML_TYPE_IQ1_M;
                else if (std::strcmp(argv[arg], "iq2_xxs") == 0) type = GGML_TYPE_IQ2_XXS;
                require(type != GGML_TYPE_COUNT, "Unsupported expert-cache parity quantization");
                weights.types = {type, type, GGML_TYPE_IQ4_NL};
            }
            else throw std::runtime_error("Unsupported expert-cache test argument");
        }
        set_environment("TS_HOST_MOE_EXPERT_CACHE_PREFETCH", prefetch ? "1" : "0");
        weights.fill(0.5f);
#ifdef TSG_GGML_USE_CUDA
        if (!cpu_only)
        {
            if (ggml_backend_cuda_get_device_count() < 1)
            {
                std::fprintf(stderr, "SKIP: CUDA expert-cache parity needs an NVIDIA device\n");
                return 77;
            }
            g_backend = ggml_backend_cuda_init(0);
            require(g_backend != nullptr, "Could not initialize CUDA backend");
            shared_budget_checks(weights);
            cuda_checks(weights);
            require(prefetch == (tsg::g_test_prefetch_calls > 0),
                "Opt-in raw expert prefetch did not match its requested mode");
            ggml_backend_free(g_backend);
            g_backend = nullptr;
            std::puts("PASS: CUDA compact expert cache, warm hits, eviction, invalidation, reload and budget");
            return 0;
        }
#else
        (void)cpu_only;
#endif
        g_backend = ggml_backend_cpu_init();
        require(g_backend != nullptr, "Could not initialize CPU backend");
        SharedLedger ledger;
        require(ledger.attach(), "Could not attach the CPU fallback budget fixture");
        fallback_checks(weights);
        std::array<float, hidden> x{}, out{};
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.6f, 0.4f};
        require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(), out.data(),
            "qwen4exp fused token span") == 0, "CPU backend incorrectly engaged the CUDA expert cache");
        require(Stats().reserved == 0 && Stats().calls == 0, "Fallback path allocated CUDA cache state");
        tsg::host_moe_expert_cache_release();
        ledger.check_empty();
        require(ledger.reserve_calls == 0 && ledger.detach(),
            "CPU fallback touched shared CUDA expert credit or prevented detach");
        ggml_backend_free(g_backend);
        g_backend = nullptr;
        std::puts("PASS: disabled/non-CUDA and unsupported expert-cache shapes fall back");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        tsg::host_moe_expert_cache_release();
        if (g_backend) { ggml_backend_free(g_backend); g_backend = nullptr; }
        return 1;
    }
}
