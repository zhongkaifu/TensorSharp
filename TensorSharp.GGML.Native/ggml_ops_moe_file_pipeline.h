// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_bounded_read_pipeline.h"
#include "ggml_ops_parallel_read_pipeline.h"
#include "ggml_ops_host_moe_file_batch.h"
#include "ggml_ops_moe_pipeline_timeline.h"
#include "ggml_ops_moe_file_pipeline_config.h"
#include "ggml_ops_moe_file_pipeline_plan.h"

namespace tsg {
// Used only under the existing serial file-MoE gate. It owns its copy backend,
// pinned ring and events; the separately budgeted full graph owns device memory.
// Trim drains both streams and returns physical host storage before its credit.
class HostMoeFilePipeline {
    std::shared_ptr<SharedCacheCharge> charge_;
    ggml_backend_buffer_t ring_ = nullptr;
    ggml_backend_t copy_ = nullptr, compute_ = nullptr;
    std::array<ggml_backend_event_t, 2> fences_{};
    std::array<bool, 2> submitted_{};
    ggml_backend_event_t gate_ready_ = nullptr, down_ready_ = nullptr;
    std::size_t slot_bytes_ = 0;
    struct Span {
        std::int64_t read_start = 0, read_end = 0;
        std::atomic<std::int64_t> copy_start{0}, copy_end{0};
    };
    static void check(std::uint64_t epoch) {
        if (g_ggml_error_count.load(std::memory_order_acquire) != epoch
            || g_backend_compute_failed.load(std::memory_order_acquire))
            throw std::runtime_error("Host MoE file pipeline backend failed");
    }
    static void mark(ggml_backend_t backend, std::atomic<std::int64_t>& stamp) {
#if defined(TSG_GGML_USE_CUDA)
        if (!moe_pipeline_stream_mark(backend, &stamp))
            throw std::runtime_error("Host MoE pipeline diagnostic marker failed");
#else
        (void)backend; (void)stamp;
        throw std::runtime_error("Host MoE CUDA pipeline is unavailable");
#endif
    }
public:
    ~HostMoeFilePipeline() { clear(); }
    std::size_t retained_bytes() const noexcept { return ring_ ? ggml_backend_buffer_get_size(ring_) : 0; }
    void drain() {
        // Either stream can be awaiting the other only through a successfully
        // recorded event. Copy never waits compute, so this order cannot cycle.
        if (copy_) sync_backend(copy_);
        if (compute_) sync_backend(compute_);
    }
    void clear() noexcept {
        drain();
        for (auto& event : fences_) { if (event) ggml_backend_event_free(event); event = nullptr; }
        if (gate_ready_) ggml_backend_event_free(gate_ready_);
        if (down_ready_) ggml_backend_event_free(down_ready_);
        gate_ready_ = down_ready_ = nullptr;
        if (ring_) ggml_backend_buffer_free(ring_);
        ring_ = nullptr; slot_bytes_ = 0; submitted_ = {};
        charge_.reset(); // The pinned payload has been freed.
        if (copy_) ggml_backend_free(copy_);
        copy_ = compute_ = nullptr;
    }
    // Unsupported pageable fallback is explicit: free it and use the existing
    // fully budgeted serial path before any file read or graph submission.
    bool prepare(ggml_backend_t compute, std::size_t full_host_payload,
        std::size_t maximum_ring_bytes = 16u << 20) {
        const auto requested = host_moe_file_pipeline_slot_bytes(full_host_payload, maximum_ring_bytes);
        if (!requested) return false;
        if (ring_ && compute_ == compute && slot_bytes_ == requested
            && ggml_backend_buffer_get_size(ring_) <= maximum_ring_bytes) return true;
        clear();
        const auto bytes = host_moe_checked_mul(requested, 2);
        auto charge = SharedCacheCharge::reserve(0, 3, bytes);
        if (!charge || !charge->is_accounted()) throw std::runtime_error("Host MoE pinned pipeline budget refused");
        const auto device = ggml_backend_get_device(compute);
        const auto host_type = ggml_backend_dev_host_buffer_type(device);
        if (!host_type) return false;
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(host_type, bytes);
        if (!buffer) throw std::bad_alloc();
        // CUDA's host allocator can return a pageable CPU buffer on pin failure.
        // Never claim asynchronous/pinned qualification for that fallback.
        if (ggml_backend_buffer_get_type(buffer) != host_type) {
            ggml_backend_buffer_free(buffer); return false;
        }
        if (ggml_backend_buffer_get_size(buffer) > maximum_ring_bytes) {
            ggml_backend_buffer_free(buffer); return false;
        }
        ring_ = buffer; charge_ = std::move(charge); compute_ = compute;
        try {
            if (!charge_->commit(ggml_backend_buffer_get_size(ring_)))
                throw std::runtime_error("Host MoE pinned pipeline commit refused");
            copy_ = ggml_backend_dev_init(device, nullptr);
            if (!copy_) throw std::runtime_error("Host MoE pipeline copy backend creation failed");
            for (auto& event : fences_) if (!(event = ggml_backend_event_new(device)))
                throw std::runtime_error("Host MoE pipeline slot fence creation failed");
            gate_ready_ = ggml_backend_event_new(device); down_ready_ = ggml_backend_event_new(device);
            if (!gate_ready_ || !down_ready_) throw std::runtime_error("Host MoE pipeline dependency fence creation failed");
            slot_bytes_ = requested;
            return true;
        } catch (...) { clear(); throw; }
    }
    template<class Pool, class BeforeRead, class Failure>
    void run(Pool& pool, const HostMoeFileBatch& batch, const std::array<ggml_tensor*, 3>& weights,
        ggml_cgraph* graph, ggml_tensor* down, int rows, bool diagnostic,
        BeforeRead before_read, Failure failure) {
        if (!ring_ || !copy_ || !compute_) throw std::runtime_error("Unprepared host MoE pipeline");
        HostMoeFilePipelinePlan plan(slot_bytes_);
        for (int part = 0; part < 3; ++part)
            if (!batch.sources[part] || batch.sources[part]->bytes() != batch.extent[part])
                throw std::runtime_error("Host MoE pipeline expert extent changed");
        batch.for_each_read_range([&](int part, auto source, auto destination, auto bytes) {
            plan.append(part, source, destination - batch.offset[part], bytes);
        });
        plan.validate(batch.bytes);
        const auto& chunks = plan.transfers();
        const auto gate_chunks = plan.gate_transfers();
        // Full-layer serial reads already use the joined pool across source
        // ranges. Let it fill each bounded slot in parallel too, instead of
        // leaving its workers blocked behind two single-reader slots.
        constexpr std::size_t maximum_read_bytes = 1u << 20;
        std::vector<std::size_t> read_counts;
        read_counts.reserve(chunks.size());
        std::size_t read_jobs = 0;
        for (std::size_t i = 0; i < chunks.size(); ++i)
        {
            const auto count = plan.read_job_count(i, maximum_read_bytes);
            read_jobs = host_moe_checked_add(read_jobs, count);
            read_counts.push_back(count);
        }
        int split = 0;
        while (split < graph->n_nodes && graph->nodes[split] != down) ++split;
        if (split == 0 || split == graph->n_nodes || !gate_chunks)
            throw std::runtime_error("Host MoE pipeline graph split is invalid");
        // Both views refer to the original nodes/tensors and full allocation.
        // All gate/up/SwiGLU fusion stays in the first view; down/reduction stays
        // in the second. No expert reduction or batch shape is reconstructed.
        auto first = ggml_graph_view(graph, 0, split);
        auto second = ggml_graph_view(graph, split, graph->n_nodes);
        if (q8_f32_execution_backend(compute_, graph) != compute_)
            throw std::runtime_error("Host MoE pipeline requires the original CUDA execution backend");
        auto spans = diagnostic ? std::make_unique<Span[]>(chunks.size()) : nullptr;
        std::atomic<std::int64_t> compute_start{0}, compute_end{0};
        const auto start = diagnostic ? moe_pipeline_clock_ns() : 0;
        const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
        auto* base = static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(ring_));
        std::size_t gate_submitted = 0;
        // This guard is declared after all timestamp storage and graph views.
        // Even the first read/transfer/compute error drains before their lifetime
        // ends. The I/O pool separately joins all producers before propagating.
        struct Drain { HostMoeFilePipeline& owner; ~Drain() { owner.drain(); } } drain{*this};
        run_bounded_parallel_read_pipeline(pool, read_counts, 2, [&](std::size_t index, std::size_t slot) {
            if (submitted_[slot]) ggml_backend_event_synchronize(fences_[slot]);
            check(epoch);
            if (spans) spans[index].read_start = moe_pipeline_clock_ns();
        }, [&](std::size_t index, std::size_t job, std::size_t slot) {
            plan.read_job(index, job, maximum_read_bytes, base + slot * slot_bytes_, before_read,
                [&](int part, auto source, auto* destination, auto bytes) {
                    batch.sources[part]->read(source, destination, bytes);
                });
        }, [&](std::size_t index, std::size_t slot) {
            if (spans) spans[index].read_end = moe_pipeline_clock_ns();
            const auto& chunk = chunks[index];
            if (spans) mark(copy_, spans[index].copy_start);
            ggml_backend_tensor_set_async(copy_, weights[chunk.part], base + slot * slot_bytes_,
                chunk.destination, chunk.bytes);
            if (spans) mark(copy_, spans[index].copy_end);
            ggml_backend_event_record(fences_[slot], copy_);
            submitted_[slot] = true;
            check(epoch); failure(1);
            if (chunk.part != 2 && ++gate_submitted == gate_chunks) {
                ggml_backend_event_record(gate_ready_, copy_);
                ggml_backend_event_wait(compute_, gate_ready_);
                if (spans) mark(compute_, compute_start);
                if (ggml_backend_graph_compute_async(compute_, &first) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("Host MoE pipeline gate/up compute failed");
                if (spans) mark(compute_, compute_end);
                check(epoch); failure(3); // Test an error while gate/up may still run.
            }
        });
        if (gate_submitted != gate_chunks) throw std::runtime_error("Host MoE pipeline gate/up submission incomplete");
        ggml_backend_event_record(down_ready_, copy_);
        ggml_backend_event_wait(compute_, down_ready_);
        if (ggml_backend_graph_compute_async(compute_, &second) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Host MoE pipeline down compute failed");
        drain.owner.drain(); check(epoch); failure(2);
        if (spans) {
            const auto a = compute_start.load(std::memory_order_acquire), b = compute_end.load(std::memory_order_acquire);
            std::int64_t read_overlap = 0, copy_overlap = 0;
            auto overlap = [&](std::int64_t x, std::int64_t y) {
                return std::max<std::int64_t>(0, std::min(b, y) - std::max(a, x));
            };
            for (std::size_t i = 0; i < chunks.size(); ++i) if (chunks[i].part == 2) {
                read_overlap += overlap(spans[i].read_start, spans[i].read_end);
                copy_overlap += overlap(spans[i].copy_start.load(), spans[i].copy_end.load());
            }
            std::fprintf(stderr, "[HOSTMOE-FILE-PIPELINE] seq=%d chunks=%zu read_jobs=%zu max_read_bytes=%zu pinned=%zu total_ms=%.3f gate_stream_start_ns=%lld gate_stream_end_ns=%lld down_read_overlap_sum_ms=%.3f down_copy_overlap_sum_ms=%.3f\n",
                rows, chunks.size(), read_jobs, maximum_read_bytes, slot_bytes_ * 2, (moe_pipeline_clock_ns()-start)/1e6,
                static_cast<long long>(a), static_cast<long long>(b), read_overlap/1e6, copy_overlap/1e6);
            for (std::size_t i = 0; i < chunks.size(); ++i)
                std::fprintf(stderr, "[HOSTMOE-FILE-PIPELINE-SPAN] seq=%d chunk=%zu part=%d bytes=%zu read_start_ns=%lld read_end_ns=%lld copy_start_ns=%lld copy_end_ns=%lld\n",
                    rows, i, chunks[i].part, chunks[i].bytes,
                    static_cast<long long>(spans[i].read_start), static_cast<long long>(spans[i].read_end),
                    static_cast<long long>(spans[i].copy_start.load()), static_cast<long long>(spans[i].copy_end.load()));
        }
    }
};
}
