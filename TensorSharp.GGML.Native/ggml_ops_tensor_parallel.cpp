// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ---------------------------------------------------------------------------
// Tensor parallelism for the GGML backend.
//
// Design (mirrors llama.cpp's `--split-mode tensor` where it applies, adapted
// to TensorSharp's host-resident tensor model):
//
//   * One ggml backend per GPU, held in tsg::g_device_states[]. Ops select a
//     rank with TSGgml_SetActiveDevice; every existing single-device op then
//     runs on that GPU unchanged (`g_backend` resolves to the active slot).
//
//   * Column-parallel weights are sliced along ne1 and row-parallel weights
//     along ne0 by the managed layer, so each rank's matmul is 1/tp of the work
//     and 1/tp of the VRAM.
//
//   * Row-parallel layers end in an AllReduce. Two implementations:
//       - device AllReduce: the per-rank partial sums stay in VRAM and are
//         reduced with ggml-cuda's comm backend (NCCL when available, P2P
//         pipeline otherwise, butterfly as a last resort). Nothing crosses PCIe
//         to the host. This is the fast path and is what the fused multi-rank
//         entry points below use.
//       - host AllReduce: the partials are already back in host RAM (the
//         generic per-op path downloads every result), so they are summed
//         in-place with an AVX/NEON-friendly loop. No device round trip.
//
//   * The fused multi-rank entry points submit one graph per rank with
//     ggml_backend_graph_compute_async and only then synchronize, so all GPUs
//     compute concurrently from a single host thread. This is the same
//     submit-all-then-sync structure ggml's meta backend uses.
// ---------------------------------------------------------------------------
#include "ggml_ops_internal.h"
#include "ggml_ops_q8_weight_policy.h"
// ggml_graph_view: the segment executor runs [i0, i1) of an already-built graph
// without copying it, exactly as ggml_backend_sched does for its splits.
#include "ggml-impl.h"

#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace tsg
{
    std::atomic<int> g_tp_global_degree{0};
    std::atomic<int> g_tp_rank_offset{0};

    // --- Communicator ------------------------------------------------------

    namespace
    {
        // Function pointers resolved from the backend registry. ggml-cuda
        // exposes them via ggml_backend_reg_get_proc_address; other backends
        // (CPU / Metal / Vulkan) do not, and fall back to the host reduction.
        using comm_init_fn = void* (*)(ggml_backend_t*, size_t);
        using comm_free_fn = void  (*)(void*);
        using comm_allreduce_fn = bool  (*)(void*, ggml_tensor**);

        struct TpComm
        {
            void* ctx = nullptr;
            comm_free_fn free_fn = nullptr;
            comm_allreduce_fn allreduce_fn = nullptr;
            bool attempted = false;
        };

        TpComm g_tp_comm;
        std::mutex g_tp_comm_mutex;

        // Set a process environment variable so that native getenv() sees it —
        // the backend and NCCL both read their configuration that way.
        void tp_setenv(const char* name, const char* value)
        {
#if defined(_WIN32)
            _putenv_s(name, value);
#else
            setenv(name, value, 1);
#endif
        }

        // Scratch device tensors used by the device AllReduce path. One
        // context+buffer per rank, grown on demand and reused across calls so a
        // steady-state decode never allocates.
        struct AllReduceScratch
        {
            ggml_context* ctx = nullptr;
            ggml_backend_buffer_t buffer = nullptr;
            ggml_tensor* tensor = nullptr;
            std::size_t capacity = 0;
        };

        AllReduceScratch g_ar_scratch[TSG_MAX_DEVICES];
        std::mutex g_ar_scratch_mutex;

        void free_ar_scratch_locked(int rank)
        {
            AllReduceScratch& s = g_ar_scratch[rank];
            if (s.buffer != nullptr)
            {
                tsg::graph_budget_free_buffer(s.buffer);
                s.buffer = nullptr;
            }
            if (s.ctx != nullptr)
            {
                ggml_free(s.ctx);
                s.ctx = nullptr;
            }
            s.tensor = nullptr;
            s.capacity = 0;
        }

        // Ensure rank `rank` has a contiguous F32 device tensor of at least
        // `count` elements. Must be called with the rank already active.
        ggml_tensor* ensure_ar_scratch_locked(int rank, std::int64_t count)
        {
            AllReduceScratch& s = g_ar_scratch[rank];
            const std::size_t need = static_cast<std::size_t>(count) * sizeof(float);
            if (s.tensor != nullptr && s.capacity >= need)
            {
                // Reinterpret the cached allocation at the requested length. The
                // buffer is large enough, so only ne/nb need adjusting.
                s.tensor->ne[0] = count;
                s.tensor->ne[1] = 1;
                s.tensor->ne[2] = 1;
                s.tensor->ne[3] = 1;
                s.tensor->nb[0] = sizeof(float);
                s.tensor->nb[1] = need;
                s.tensor->nb[2] = need;
                s.tensor->nb[3] = need;
                return s.tensor;
            }

            free_ar_scratch_locked(rank);

            ggml_init_params params = {};
            params.mem_size = ggml_tensor_overhead() * 2;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            s.ctx = ggml_init(params);
            if (s.ctx == nullptr)
                return nullptr;

            // Round up so a slowly growing prefill does not reallocate per call.
            const std::size_t slab = 1u << 20;
            const std::size_t alloc = ((need + slab - 1) / slab) * slab;
            const std::int64_t alloc_count = static_cast<std::int64_t>(alloc / sizeof(float));

            s.tensor = ggml_new_tensor_1d(s.ctx, GGML_TYPE_F32, alloc_count);
            if (s.tensor == nullptr)
            {
                free_ar_scratch_locked(rank);
                return nullptr;
            }
            s.buffer = tsg::alloc_ctx_tensors_budgeted(s.ctx, g_backend);
            if (s.buffer == nullptr)
            {
                free_ar_scratch_locked(rank);
                return nullptr;
            }
            s.capacity = alloc;
            if (vram_log_enabled())
                vram_log("tp-allreduce-scratch", static_cast<std::int64_t>(alloc));

            s.tensor->ne[0] = count;
            s.tensor->nb[1] = need;
            s.tensor->nb[2] = need;
            s.tensor->nb[3] = need;
            return s.tensor;
        }
    } // namespace

    // Resolve (once) the backend's collective-communication entry points for the
    // currently initialized rank set. Returns false when the backend has none.
    bool tp_comm_ensure()
    {
        std::lock_guard<std::mutex> lock(g_tp_comm_mutex);
        if (g_tp_comm.attempted)
            return g_tp_comm.ctx != nullptr;
        g_tp_comm.attempted = true;

        const int n = g_device_count.load(std::memory_order_acquire);
        if (n <= 1)
            return false;

        ggml_backend_t backend0 = dev(0).backend;
        if (backend0 == nullptr)
            return false;

        ggml_backend_dev_t d0 = ggml_backend_get_device(backend0);
        ggml_backend_reg_t reg = d0 != nullptr ? ggml_backend_dev_backend_reg(d0) : nullptr;
        if (reg == nullptr)
            return false;

#ifdef TSG_GGML_USE_CUDA
        // Pick a collective that actually works on THIS host, for THIS rank
        // count, before the backend creates its communicator.
        //
        // The CUDA comm chain is nccl -> internal -> none, and each link has a
        // way of failing quietly:
        //   * NCCL trusts the driver's P2P capability report. Where that report
        //     is a lie (virtualized cloud boxes), init succeeds and the FIRST
        //     collective wedges every GPU in peer-waiting spin kernels.
        //   * the "internal" pinned-host pipeline only supports exactly 2
        //     devices, so at 3+ ranks it declines and the chain lands on
        //     "none", whose AllReduce always returns false. Every segment
        //     boundary then drains its partials to host RAM and pushes the sum
        //     back — a correct result at a fraction of the speed.
        // So a host with false P2P used to lose the device collective entirely
        // the moment it had more than two GPUs.
        //
        // Instead: check peer traffic behaviourally first (no NCCL needed), and
        // when it is broken, take P2P away from NCCL rather than taking NCCL
        // away from the model. NCCL then routes over its shared-memory
        // transport, which keeps the reduction off the per-segment host path and
        // works for any rank count the backend supports. The internal pipeline
        // stays as the 2-device fallback for when NCCL is unusable outright.
        // An explicit GGML_CUDA_ALLREDUCE from the operator wins unconditionally.
        {
            const char* reg_name = ggml_backend_reg_name(reg);
            if (reg_name != nullptr && std::strcmp(reg_name, "CUDA") == 0
                && std::getenv("GGML_CUDA_ALLREDUCE") == nullptr)
            {
                int probe_devices[TSG_MAX_DEVICES];
                for (int r = 0; r < n; ++r)
                    probe_devices[r] = dev(r).device_index >= 0 ? dev(r).device_index : r;

                // NCCL caches its tunables at the first communicator init in the
                // process, so this has to happen before any probe touches NCCL.
                if (std::getenv("NCCL_P2P_DISABLE") == nullptr
                    && tp_probe_cuda_peer_access(probe_devices, n) == 0)
                {
                    tp_setenv("NCCL_P2P_DISABLE", "1");
                    std::fprintf(stderr,
                        "[TP] peer access is advertised but non-functional on this host; "
                        "running NCCL with NCCL_P2P_DISABLE=1 (shared-memory transport).\n");
                    std::fflush(stderr);
                }

                if (tp_probe_cuda_collective(probe_devices, n) == 0)
                {
                    // NCCL is unusable even without P2P. The internal pipeline
                    // is 2-device only; past that the host reduction is all that
                    // is left, and saying so beats a silent 2x.
                    tp_setenv("GGML_CUDA_ALLREDUCE", n == 2 ? "internal" : "none");
                    std::fprintf(stderr,
                        n == 2
                        ? "[TP] device collective (NCCL) is non-functional on this host; "
                          "using the pinned-host-memory AllReduce instead. Set "
                          "GGML_CUDA_ALLREDUCE to override, TS_GGML_TP_AR_PROBE=force to re-test.\n"
                        : "[TP] device collective (NCCL) is non-functional on this host and the "
                          "pinned-host-memory AllReduce only supports 2 devices; falling back to "
                          "the host reduction, which will cost throughput. Set "
                          "GGML_CUDA_ALLREDUCE to override, TS_GGML_TP_AR_PROBE=force to re-test.\n");
                    std::fflush(stderr);
                }
            }
        }
#endif

        auto init_fn = reinterpret_cast<comm_init_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init"));
        auto free_fn = reinterpret_cast<comm_free_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free"));
        auto allreduce_fn = reinterpret_cast<comm_allreduce_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor"));
        if (init_fn == nullptr || allreduce_fn == nullptr)
            return false;

        std::vector<ggml_backend_t> backends;
        backends.reserve(static_cast<std::size_t>(n));
        for (int r = 0; r < n; ++r)
        {
            if (dev(r).backend == nullptr)
                return false;
            backends.push_back(dev(r).backend);
        }

        g_tp_comm.ctx = init_fn(backends.data(), backends.size());
        g_tp_comm.free_fn = free_fn;
        g_tp_comm.allreduce_fn = allreduce_fn;
        return g_tp_comm.ctx != nullptr;
    }

    void tp_comm_free()
    {
        {
            std::lock_guard<std::mutex> lock(g_ar_scratch_mutex);
            for (int r = 0; r < TSG_MAX_DEVICES; ++r)
            {
                if (g_ar_scratch[r].ctx == nullptr && g_ar_scratch[r].buffer == nullptr)
                    continue;
                ScopedRank rank(r);
                free_ar_scratch_locked(r);
            }
        }

        std::lock_guard<std::mutex> lock(g_tp_comm_mutex);
#ifdef TSG_GGML_USE_CUDA
        tp_cuda_allreduce_f32_free();
#endif
        if (g_tp_comm.ctx != nullptr && g_tp_comm.free_fn != nullptr)
            g_tp_comm.free_fn(g_tp_comm.ctx);
        g_tp_comm.ctx = nullptr;
        g_tp_comm.free_fn = nullptr;
        g_tp_comm.allreduce_fn = nullptr;
        g_tp_comm.attempted = false;
    }

    // Whether the backend's collective can actually reduce, verified once by
    // running a real one-element AllReduce over the per-rank scratch tensors.
    // `tp_comm_ensure()` alone cannot answer this: the CUDA chain's last link
    // hands back a context whose AllReduce unconditionally declines, so a
    // "device collective" that silently reduces through host RAM looks
    // identical to a working one until throughput is measured.
    bool tp_device_allreduce_usable()
    {
        static std::mutex probe_mutex;
        static int cached = -1;

        std::lock_guard<std::mutex> lock(probe_mutex);
        if (cached >= 0)
            return cached == 1;

        const int n = g_device_count.load(std::memory_order_acquire);
        if (n <= 1 || !tp_comm_ensure())
        {
            cached = 0;
            return false;
        }

        // 32 floats, not 1: the internal pipeline asserts that the reduced byte
        // count is a multiple of 16, so a single-element probe aborts the
        // process instead of answering the question.
        constexpr std::int64_t probe_elems = 32;

        ggml_tensor* probe[TSG_MAX_DEVICES] = {};
        {
            std::lock_guard<std::mutex> scratch_lock(g_ar_scratch_mutex);
            for (int r = 0; r < n; ++r)
            {
                ScopedRank rank(r);
                probe[r] = ensure_ar_scratch_locked(r, probe_elems);
                if (probe[r] == nullptr)
                {
                    cached = 0;
                    return false;
                }
            }
        }

        bool ok;
        {
            std::lock_guard<std::mutex> comm_lock(g_tp_comm_mutex);
            ok = g_tp_comm.ctx != nullptr && g_tp_comm.allreduce_fn != nullptr
                && g_tp_comm.allreduce_fn(g_tp_comm.ctx, probe);
        }
        if (ok)
        {
            for (int r = 0; r < n; ++r)
            {
                ScopedRank rank(r);
                tsg::sync_backend(g_backend);
            }
        }
        cached = ok ? 1 : 0;
        return ok;
    }

    // In-place AllReduce over device tensors, one per rank. `tensors[r]` must be
    // contiguous F32 and resident on rank r's backend. Returns false when the
    // backend has no collective support (caller falls back to the host sum).
    bool tp_device_allreduce(ggml_tensor** tensors)
    {
        if (!tp_comm_ensure())
            return false;
        std::lock_guard<std::mutex> lock(g_tp_comm_mutex);
        if (g_tp_comm.ctx == nullptr)
            return false;
        return g_tp_comm.allreduce_fn(g_tp_comm.ctx, tensors);
    }

    // Host-side AllReduce: sum n contiguous F32 buffers element-wise and write
    // the total back into every buffer. Compilers vectorize both loops; the
    // reduction is memory-bandwidth bound and, for TensorSharp's host-resident
    // tensors, strictly cheaper than staging through VRAM.
    void tp_host_allreduce(float** buffers, int n, std::int64_t count)
    {
        if (n <= 1 || count <= 0)
            return;

        float* dst = buffers[0];
        for (int r = 1; r < n; ++r)
        {
            const float* src = buffers[r];
            for (std::int64_t i = 0; i < count; ++i)
                dst[i] += src[i];
        }
        const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
        for (int r = 1; r < n; ++r)
            std::memcpy(buffers[r], dst, bytes);
    }

    // Multi-threaded variant for the large buffers a long prefill produces.
    // Splitting by element range keeps every thread on its own cache lines.
    void tp_host_allreduce_mt(float** buffers, int n, std::int64_t count)
    {
        // Below this the thread hand-off costs more than the sum itself.
        constexpr std::int64_t k_parallel_threshold = 1 << 16; // 256 KB of F32
        if (n <= 1 || count <= 0)
            return;
        if (count < k_parallel_threshold)
        {
            tp_host_allreduce(buffers, n, count);
            return;
        }

        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4;
        int threads = static_cast<int>(hw > 8 ? 8 : hw);
        const std::int64_t chunk = (count + threads - 1) / threads;

        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(threads - 1));
        auto work = [&](std::int64_t begin, std::int64_t end)
        {
            float* dst = buffers[0];
            for (int r = 1; r < n; ++r)
            {
                const float* src = buffers[r];
                for (std::int64_t i = begin; i < end; ++i)
                    dst[i] += src[i];
            }
            for (int r = 1; r < n; ++r)
                std::memcpy(buffers[r] + begin, dst + begin,
                    static_cast<std::size_t>(end - begin) * sizeof(float));
        };

        for (int t = 1; t < threads; ++t)
        {
            const std::int64_t begin = chunk * t;
            const std::int64_t end = (begin + chunk < count) ? begin + chunk : count;
            if (begin >= end) break;
            pool.emplace_back(work, begin, end);
        }
        work(0, chunk < count ? chunk : count);
        for (auto& th : pool)
            th.join();
    }

    // --- Fused tensor-parallel graph execution ------------------------------

    bool tp_fused_available(int rank_count, bool distributed)
    {
        // A distributed run may legitimately drive ONE local rank per node:
        // the fused graph still collapses the per-op dispatches into a single
        // graph per token, and the reduction that a second local rank would
        // have provided happens across nodes instead.
        if (rank_count < (distributed ? 1 : 2))
            return false;
        if (rank_count != g_device_count.load(std::memory_order_acquire))
            return false;
        // Resolve the device collective (NCCL / P2P) when the backend has one;
        // without it the segment boundaries reduce through host staging in
        // tp_reduce_segment instead. That is still far cheaper than the per-op
        // fallback (two host crossings per layer instead of ~30), and it is
        // the only transport on backends with no comm support (Vulkan).
        tp_comm_ensure();
        return true;
    }

    bool tp_plan_segments(TpRankPlan& plan, const std::vector<ggml_tensor*>& boundary)
    {
        plan.seg_end.clear();
        plan.host_moe_at.clear();
        if (plan.graph == nullptr)
            return false;

        const int n_nodes = ggml_graph_n_nodes(plan.graph);
        // Index the node list once, then resolve each boundary against it. The
        // cut order is the GRAPH's, not the order the builder recorded them in:
        // those agree only while every node is emitted by one trailing
        // build_forward_expand. A kernel that expands anything early -- an
        // offloaded layer's MoE seam, say -- pulls part of a later layer's chain
        // forward and the two orders diverge, which is why this cannot be a
        // single forward cursor. Sorting by node index is correct regardless:
        // each partial is reduced immediately after the node that completes it.
        std::unordered_map<const ggml_tensor*, int> node_index;
        node_index.reserve(static_cast<std::size_t>(n_nodes) * 2);
        for (int i = 0; i < n_nodes; ++i)
            node_index.emplace(plan.graph->nodes[i], i);

        std::vector<std::pair<int, ggml_tensor*>> ar_at;
        ar_at.reserve(boundary.size());
        for (std::size_t b = 0; b < boundary.size(); ++b)
        {
            auto it = node_index.find(boundary[b]);
            if (it == node_index.end())
            {
                set_last_error("Tensor-parallel plan: a segment boundary tensor is not a node of the built graph.");
                return false;
            }
            ar_at.emplace_back(it->second + 1,
                b < plan.ar_tensor.size() ? plan.ar_tensor[b] : nullptr);
        }
        std::sort(ar_at.begin(), ar_at.end(),
            [](const std::pair<int, ggml_tensor*>& a, const std::pair<int, ggml_tensor*>& b) { return a.first < b.first; });

        std::vector<int> ar_cut;
        std::vector<ggml_tensor*> ar_in;
        ar_cut.reserve(ar_at.size());
        ar_in.reserve(ar_at.size());
        for (const auto& e : ar_at)
        {
            // Two partials completing on the same node would need two reductions
            // at one cut; the schedule has room for exactly one, so keep the
            // first and let the second share its boundary (it cannot be later).
            if (!ar_cut.empty() && ar_cut.back() == e.first)
                continue;
            ar_cut.push_back(e.first);
            ar_in.push_back(e.second);
        }

        // MoE CPU offload adds its own pauses (see HostMoeSegment). Merge the
        // two cut lists into ONE schedule rather than nesting them: an offloaded
        // layer's seam sits between the AllReduce that completes the previous
        // layer and the one that completes this layer's FFN, so both kinds of
        // boundary are just points where the segment loop stops.
        std::vector<int> hm_cut;
        if (!plan.host_moe.empty())
        {
            // The trailing n_nodes entry this appends is dropped below; the
            // merge re-adds it once.
            if (!host_moe_build_segment_ends(plan.graph, plan.host_moe, hm_cut, "Tensor-parallel plan"))
                return false;
            if (!hm_cut.empty())
                hm_cut.pop_back();
        }

        plan.ar_tensor.clear();
        std::size_t ia = 0, ih = 0;
        while (ia < ar_cut.size() || ih < hm_cut.size())
        {
            const int cut = (ih == hm_cut.size()) ? ar_cut[ia]
                          : (ia == ar_cut.size()) ? hm_cut[ih]
                          : (ar_cut[ia] < hm_cut[ih] ? ar_cut[ia] : hm_cut[ih]);
            // A seam and a reduction can only land on the same node if the
            // builder recorded both for one tensor; keep them on one boundary
            // (the host writes moe_out, then the collective reduces) instead of
            // emitting a zero-length segment between them.
            ggml_tensor* ar = nullptr;
            int hm = -1;
            if (ia < ar_cut.size() && ar_cut[ia] == cut)
            {
                ar = ia < ar_in.size() ? ar_in[ia] : nullptr;
                ++ia;
            }
            if (ih < hm_cut.size() && hm_cut[ih] == cut) { hm = static_cast<int>(ih); ++ih; }
            plan.seg_end.push_back(cut);
            plan.ar_tensor.push_back(ar);
            plan.host_moe_at.push_back(hm);
        }

        // Trailing segment: everything after the last boundary (final norm, LM
        // head, the output copy). Pushed unconditionally so there is always
        // exactly one more segment than there are actions — when the last
        // boundary IS the final node the tail is simply empty, and the loop
        // skips it while still performing that boundary's action.
        plan.seg_end.push_back(n_nodes);
        return true;
    }

    namespace
    {
        // Persistent per-rank staging for the host segment reduction. Reused
        // across boundaries so a prefill-sized partial (tens of MB) is not
        // re-allocated and zero-filled ~100 times per forward. Only the
        // single driving thread of tp_execute_plans grows these; the workers
        // below touch disjoint ranks.
        std::vector<float> g_reduce_staging[TSG_MAX_DEVICES];

        // Long-lived workers that fan one per-rank staging job out across the
        // ranks above 0 (rank 0 runs on the caller). A segment boundary fires
        // twice per layer per token, so the ~50 µs of a thread spawn matters;
        // a condvar wake is an order of magnitude cheaper and lets the ranks'
        // fence waits overlap. Worker r is permanently bound to rank r+1.
        // The pool is leaked deliberately: workers park on the condvar and die
        // with the process, avoiding static-destruction races with the ggml
        // backends they never touch while idle.
        class TpStagePool
        {
        public:
            using Job = std::function<void(int)>;

            void run(int rank_count, const Job& job)
            {
                const int extra = rank_count - 1;
                if (extra <= 0)
                {
                    job(0);
                    return;
                }
                ensure(extra);
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _job = &job;
                    _active = extra;
                    _pending = extra;
                    ++_generation;
                }
                _wake.notify_all();
                job(0);
                std::unique_lock<std::mutex> lock(_mutex);
                _done.wait(lock, [&] { return _pending == 0; });
                _job = nullptr;
            }

        private:
            void ensure(int extra)
            {
                while (static_cast<int>(_workers.size()) < extra)
                {
                    const int rank = static_cast<int>(_workers.size()) + 1;
                    _workers.emplace_back([this, rank] { worker(rank); });
                }
            }

            void worker(int rank)
            {
                std::uint64_t seen = 0;
                for (;;)
                {
                    const Job* job = nullptr;
                    {
                        std::unique_lock<std::mutex> lock(_mutex);
                        _wake.wait(lock, [&] { return _generation != seen && _job != nullptr && rank <= _active; });
                        seen = _generation;
                        job = _job;
                    }
                    (*job)(rank);
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        if (--_pending == 0)
                            _done.notify_all();
                    }
                }
            }

            std::mutex _mutex;
            std::condition_variable _wake;
            std::condition_variable _done;
            std::vector<std::thread> _workers;
            const Job* _job = nullptr;
            int _active = 0;
            int _pending = 0;
            std::uint64_t _generation = 0;
        };

        TpStagePool& tp_stage_pool()
        {
            static TpStagePool* pool = new TpStagePool();
            return *pool;
        }

        // Reduce one segment's partials. Returns false only when the collective
        // is genuinely unusable — a per-call refusal (an unsupported shape, say)
        // falls back to summing through the host so the token still completes.
        bool tp_reduce_segment(ggml_tensor** nodes, int rank_count,
                               bool exact_f32 = false, bool host_only = false)
        {
            if (exact_f32)
            {
                // Upstream NCCL silently converts >=32768 F32 elements to BF16
                // for two ranks (higher thresholds for more ranks). Its NCCL
                // branch ignores GGML_CUDA_AR_BF16_THRESHOLD. Use zero-copy tensor
                // views below that threshold, preserving the original device
                // buffers and streams. No upstream patch or weight conversion.
                // Also respect the internal transport's configurable threshold;
                // if it compresses even one float, use our F32 host reduction.
                for (int r = 0; r < rank_count; ++r)
                    if (nodes[r]->type != GGML_TYPE_F32 || !ggml_is_contiguous(nodes[r])
                        || ggml_nelements(nodes[r]) != ggml_nelements(nodes[0]))
                    {
                        set_last_error("Exact tensor-parallel reduction requires equal contiguous F32 partials.");
                        return false;
                    }
#ifdef TSG_GGML_USE_CUDA
                // Reuse the CUDA rank streams and the already-probed transport
                // selection, but own our NCCL communicator so precision is explicit.
                // A missing NCCL runtime retains the exact chunked/host fallback.
                if (tp_comm_ensure())
                {
                    ggml_backend_t backends[TSG_MAX_DEVICES];
                    for (int r = 0; r < rank_count; ++r) backends[r] = dev(r).backend;
                    std::lock_guard<std::mutex> lock(g_tp_comm_mutex);
                    const int result = tp_cuda_allreduce_f32(backends, nodes, rank_count);
                    if (result != 0) return result > 0;
                }
#endif
                // The internal CUDA transport requires 16-byte aligned data
                // and lengths. Keep each next view aligned as well as staying
                // below NCCL's smallest compression threshold.
                int64_t chunk_limit = 32764;
                const char* threshold = std::getenv("GGML_CUDA_AR_BF16_THRESHOLD");
                if (threshold && threshold[0])
                {
                    const unsigned long long bytes = std::strtoull(threshold, nullptr, 10);
                    if (bytes > 0) chunk_limit = std::min<int64_t>(chunk_limit, (bytes - 1) / sizeof(float));
                }
                chunk_limit -= chunk_limit % 4;
                if (chunk_limit > 0)
                {
                    const int64_t count = ggml_nelements(nodes[0]);
                    ggml_tensor views[TSG_MAX_DEVICES];
                    ggml_tensor* pointers[TSG_MAX_DEVICES];
                    for (int64_t start = 0; start < count; start += chunk_limit)
                    {
                        const int64_t length = std::min(chunk_limit, count - start);
                        for (int r = 0; r < rank_count; ++r)
                        {
                            views[r] = *nodes[r];
                            views[r].ne[0] = length;
                            views[r].ne[1] = views[r].ne[2] = views[r].ne[3] = 1;
                            views[r].nb[0] = sizeof(float);
                            views[r].nb[1] = views[r].nb[2] = views[r].nb[3] = length * sizeof(float);
                            views[r].data = static_cast<char*>(nodes[r]->data) + start * sizeof(float);
                            views[r].view_src = nodes[r];
                            views[r].view_offs = start * sizeof(float);
                            pointers[r] = &views[r];
                        }
                        bool device_compatible = length % 4 == 0;
                        for (int r = 0; r < rank_count; ++r)
                            device_compatible &= reinterpret_cast<uintptr_t>(views[r].data) % 16 == 0;
                        if ((!device_compatible || !tp_device_allreduce(pointers))
                            && !tp_reduce_segment(pointers, rank_count, false, true))
                            return false;
                    }
                    return true;
                }
                host_only = true;
            }
            if (!host_only && tp_device_allreduce(nodes))
                return true;

            // Host reduction: drain every rank, sum the partials in RAM, push
            // the total back. On backends with no device collective (Vulkan)
            // this is the routine transport for every segment boundary, so the
            // per-rank staging runs concurrently — each rank owns its own
            // backend queue and g_active_rank is thread-local, and the fence
            // waits overlap instead of chaining.
            const std::int64_t count = ggml_nelements(nodes[0]);
            if (count <= 0)
                return true;
            if (nodes[0]->type != GGML_TYPE_F32)
            {
                set_last_error("Tensor-parallel segment reduction requires F32 partials.");
                return false;
            }

            const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
            float* ptrs[TSG_MAX_DEVICES] = {};
            for (int r = 0; r < rank_count; ++r)
            {
                auto& stage = g_reduce_staging[r];
                if (stage.size() < static_cast<std::size_t>(count))
                    stage.resize(static_cast<std::size_t>(count));
                ptrs[r] = stage.data();
            }

            auto stage_down = [&](int r)
            {
                ScopedRank rank(r);
                tsg::sync_backend(g_backend);
                ggml_backend_tensor_get(nodes[r], ptrs[r], 0, bytes);
            };
            auto stage_up = [&](int r)
            {
                ScopedRank rank(r);
                ggml_backend_tensor_set(nodes[r], ptrs[r], 0, bytes);
            };

            // Concurrent staging pays off once the copies outweigh the wake:
            // ggml-vulkan serializes submissions behind one queue mutex, so for
            // decode-sized partials two threads only add contention (measured
            // slower than the serial loop), while prefill-sized ones overlap
            // their PCIe time.
            if (rank_count > 1 && bytes >= (256u << 10))
            {
                tp_stage_pool().run(rank_count, stage_down);
                tp_host_allreduce_mt(ptrs, rank_count, count);
                tp_stage_pool().run(rank_count, stage_up);
            }
            else
            {
                for (int r = 0; r < rank_count; ++r)
                    stage_down(r);
                tp_host_allreduce_mt(ptrs, rank_count, count);
                for (int r = 0; r < rank_count; ++r)
                    stage_up(r);
            }
            return true;
        }
    } // namespace

    bool tp_execute_plans(TpRankPlan** plans, int rank_count,
                          TpDistributedAllReduce cross_node, void* cross_node_user)
    {
        if (plans == nullptr || rank_count < 1 || rank_count > TSG_MAX_DEVICES)
        {
            set_last_error("Invalid tensor-parallel execution plan set.");
            return false;
        }

        const std::size_t n_seg = plans[0]->seg_end.size();
        const std::size_t n_host_moe = plans[0]->host_moe.size();
        for (int r = 0; r < rank_count; ++r)
        {
            if (plans[r] == nullptr || !plans[r]->valid() || plans[r]->seg_end.size() != n_seg ||
                plans[r]->ar_tensor.size() + 1 != n_seg ||
                plans[r]->allreduce_f32 != plans[0]->allreduce_f32 ||
                plans[r]->host_moe.size() != n_host_moe ||
                (n_host_moe != 0 && plans[r]->host_moe_at.size() + 1 != n_seg))
            {
                set_last_error("Tensor-parallel execution plans disagree on their segment structure.");
                return false;
            }
        }

        std::vector<ggml_tensor*> nodes(static_cast<std::size_t>(rank_count), nullptr);
        // Per-rank cursors: the segments before the last AllReduce are structurally
        // identical across ranks, but the trailing one is not — rank 0 may fold the
        // final norm and LM head that the other ranks do not carry.
        std::vector<int> start(static_cast<std::size_t>(rank_count), 0);

        for (std::size_t s = 0; s < n_seg; ++s)
        {
            // Submit segment s on every rank without waiting: the GPUs overlap,
            // and (because this all happens on one thread) ggml-cuda's graph
            // capture stays valid, unlike a worker-per-rank fan-out.
            for (int r = 0; r < rank_count; ++r)
            {
                ScopedRank rank(r);
                const int end = plans[r]->seg_end[s];
                const int begin = start[static_cast<std::size_t>(r)];
                start[static_cast<std::size_t>(r)] = end;
                if (end <= begin)
                    continue;
                ggml_cgraph view = ggml_graph_view(plans[r]->graph, begin, end);
                if (ggml_backend_graph_compute_async(q8_f32_execution_backend(plans[r]->backend ? plans[r]->backend : g_backend, &view), &view) != GGML_STATUS_SUCCESS)
                {
                    set_last_error("Tensor-parallel segment execution failed.");
                    return false;
                }
            }

            if (s + 1 >= n_seg)
                continue;

            // MoE CPU offload seam. The offloaded layer is computed ONCE, from
            // rank 0's copy of the inputs: the router and the residual stream it
            // reads are replicated across ranks (the previous boundary's
            // AllReduce made them so), and the host expert backend is a single
            // mutex-guarded thread pool, so splitting the matmul per rank would
            // serialize anyway while paying an extra download each.
            const int hm_idx = (n_host_moe != 0) ? plans[0]->host_moe_at[s] : -1;
            if (hm_idx >= 0)
            {
                static thread_local std::vector<float> s_moe_out;
                // Every rank has to land before any moe_out is written: the
                // segments were submitted async, and a rank still in flight
                // could otherwise see the upload out of order.
                for (int r = 0; r < rank_count; ++r)
                {
                    ScopedRank rank(r);
                    tsg::sync_backend(g_backend);
                }
                {
                    ScopedRank rank(0);
                    // allow_device_stream=false: the offloaded layer is evaluated
                    // ONCE here, on the host, over the unsharded expert stack,
                    // and the ranks differ only in where that one result lands
                    // (tp_reduced, below). Streaming it to rank 0's accelerator
                    // instead would put a whole layer of experts back on one
                    // device while the other ranks idle.
                    if (!host_moe_compute_segment(plans[0]->host_moe[static_cast<std::size_t>(hm_idx)],
                                                  s_moe_out, "Tensor-parallel host MoE",
                                                  nullptr, /*allow_device_stream*/ false))
                        return false;
                }
                for (int r = 0; r < rank_count; ++r)
                {
                    ScopedRank rank(r);
                    const HostMoeSegment& hm = plans[r]->host_moe[static_cast<std::size_t>(hm_idx)];
                    // tp_reduced: the graph still sums this layer's MoE output
                    // across ranks, so only rank 0 may carry it.
                    if (hm.tp_reduced != 0 && r != 0)
                        host_moe_zero_segment(hm);
                    else
                        host_moe_upload_segment(hm, s_moe_out.data());
                }
            }

            if (plans[0]->ar_tensor[s] != nullptr)
            {
                for (int r = 0; r < rank_count; ++r)
                    nodes[static_cast<std::size_t>(r)] = plans[r]->ar_tensor[s];
                // One local rank has nothing to reduce on-node; the cross-node
                // exchange below is the whole reduction in that configuration.
                if (rank_count > 1 && !tp_reduce_segment(nodes.data(), rank_count, plans[0]->allreduce_f32))
                    return false;

                // Multi-node: the local reduce above collapsed this node's
                // ranks. The partial is still per-NODE, so hand it to the
                // caller's cluster AllReduce and broadcast the result back
                // to every local rank before the next segment reads it.
                if (cross_node != nullptr)
                {
                    ggml_tensor* ar0 = plans[0]->ar_tensor[s];
                    const std::size_t n = static_cast<std::size_t>(ggml_nelements(ar0));
                    if (ar0->type != GGML_TYPE_F32)
                    {
                        set_last_error("Distributed tensor-parallel AllReduce requires F32 partials.");
                        return false;
                    }
                    static thread_local std::vector<float> s_xnode;
                    if (s_xnode.size() < n) s_xnode.resize(n);
                    {
                        ScopedRank rank(0);
                        ggml_backend_synchronize(g_backend);
                        ggml_backend_tensor_get(ar0, s_xnode.data(), 0, n * sizeof(float));
                    }
                    if (!cross_node(cross_node_user, s_xnode.data(), static_cast<int>(n)))
                    {
                        set_last_error("Distributed tensor-parallel AllReduce callback failed.");
                        return false;
                    }
                    for (int r = 0; r < rank_count; ++r)
                    {
                        ScopedRank rank(r);
                        ggml_backend_tensor_set(plans[r]->ar_tensor[s], s_xnode.data(), 0, n * sizeof(float));
                    }
                }
            }
        }

        for (int r = 0; r < rank_count; ++r)
        {
            ScopedRank rank(r);
            tsg::sync_backend(g_backend);
        }

        for (int r = 0; r < rank_count; ++r)
        {
            auto* p = plans[r];
            const bool has_out = p->out_tensor != nullptr && p->out_host != nullptr && p->out_bytes != 0;
            if (!has_out && p->extra_out.empty())
                continue;
            ScopedRank rank(r);
            if (has_out)
                ggml_backend_tensor_get(p->out_tensor, p->out_host, 0, p->out_bytes);
            for (const auto& d : p->extra_out)
                if (d.tensor != nullptr && d.host != nullptr && d.bytes != 0)
                    ggml_backend_tensor_get(d.tensor, d.host, 0, d.bytes);
        }
        return true;
    }
} // namespace tsg

// ---------------------------------------------------------------------------
// Fused multi-rank linear layers
//
// The whole point of tensor parallelism is that the ranks compute *at the same
// time*. Driving them through the ordinary one-op-at-a-time bridge would
// serialize them (every op ends in a synchronize + device->host copy), so the
// per-rank matmuls of a column-/row-parallel layer are submitted here as N
// asynchronous graphs and only synchronized once they are all in flight.
// ---------------------------------------------------------------------------
namespace tsg
{
    namespace
    {
        // Everything one rank needs to keep alive from graph build to sync.
        struct RankGraph
        {
            PooledContextHandle context;
            std::vector<BufferHandle> host_ptr_buffers;
            BufferHandle owned_buffer{ nullptr };
            std::vector<float> packed_input;
            TensorBinding result_binding{};
            ggml_tensor* result_node = nullptr;
            bool zero_copy_result = false;
            void* result_host = nullptr;
            std::size_t result_bytes = 0;
        };

        // Build (but do not run) `result = input @ weight` for the active rank.
        // `weight` is a ggml-quantized matrix with ne0 = inDim, ne1 = outDim.
        bool build_rank_matmul(
            RankGraph& rg,
            const TensorView2DDesc& result_desc,
            const TensorView2DDesc& input_desc,
            const QuantizedWeightDesc& weight,
            ggml_cgraph*& out_graph,
            bool keep_result_on_device)
        {
            if (!validate_desc(result_desc, "result") || !validate_desc(input_desc, "input"))
                return false;
            if (weight.data == nullptr || weight.ne0 <= 0 || weight.ne1 <= 0 || weight.raw_bytes <= 0)
            {
                set_last_error("Invalid quantized weight descriptor in the multi-rank matmul.");
                return false;
            }
            if (input_desc.dim1 != weight.ne0 || result_desc.dim1 != weight.ne1 ||
                input_desc.dim0 != result_desc.dim0)
            {
                set_last_error("Shape mismatch in the multi-rank matmul.");
                return false;
            }

            const std::size_t ctx_size = 1024 * 1024;
            if (!rg.context.init(ctx_size))
            {
                set_last_error("Failed to create a ggml context for the multi-rank matmul.");
                return false;
            }
            ggml_context* ctx = rg.context.value;

            // Try to bind both the input and the result straight onto the
            // caller's host memory (a device-resident copy cached by pointer on
            // discrete GPUs, a true zero-copy map on unified memory). When the
            // result never leaves the GPU — row-parallel with a device AllReduce
            // — only the input needs a binding.
            TensorBinding input_binding{};
            bool zero_copy = false;

            if (can_map_standard_view(input_desc))
            {
                ggml_backend_buffer_t input_buf = nullptr;
                ggml_backend_buffer_t result_buf = nullptr;
                TensorBinding ib{}, rb{};

                bool ok = create_binding_from_host_ptr_2d(ctx, g_backend, input_desc, ib, input_buf);
                if (ok && !keep_result_on_device)
                    ok = create_binding_from_host_ptr_2d(ctx, g_backend, result_desc, rb, result_buf);

                if (ok)
                {
                    rg.host_ptr_buffers.emplace_back(input_buf);
                    input_binding = ib;
                    if (!keep_result_on_device)
                    {
                        rg.host_ptr_buffers.emplace_back(result_buf);
                        rg.result_binding = rb;
                    }
                    zero_copy = true;
                }
                else
                {
                    if (result_buf != nullptr) tsg::graph_budget_free_buffer(result_buf);
                    if (input_buf != nullptr) tsg::graph_budget_free_buffer(input_buf);
                }
            }

            if (!zero_copy)
            {
                input_binding = can_map_standard_view(input_desc)
                    ? create_standard_binding(ctx, input_desc)
                    : create_packed_standard_binding(ctx, input_desc, rg.packed_input);
                if (!keep_result_on_device)
                    rg.result_binding = create_standard_binding(ctx, result_desc);
            }

            ggml_type qtype = static_cast<ggml_type>(weight.ggml_type);
            ggml_tensor* w_tensor = ggml_new_tensor_2d(ctx, qtype, weight.ne0, weight.ne1);
            if (w_tensor == nullptr || input_binding.tensor == nullptr ||
                (!keep_result_on_device && rg.result_binding.tensor == nullptr))
            {
                set_last_error("Failed to allocate ggml tensors for the multi-rank matmul.");
                return false;
            }

            // Bind the weight to its cached device copy when one exists (weights
            // are stable across calls, so this is a hit after the first token).
            bool w_bound = false;
            bool w_needs_upload = false;
            {
                ggml_backend_dev_t d = ggml_backend_get_device(g_backend);
                if (d != nullptr && weight.raw_bytes >= 4096)
                {
                    ggml_backend_buffer_t buf = nullptr;
                    void* addr = nullptr;
                    if (try_get_cacheable_tensor_buffer(
                            g_backend, d, w_tensor, weight.data,
                            static_cast<std::size_t>(weight.raw_bytes), buf, addr, w_needs_upload))
                    {
                        w_bound = ggml_backend_tensor_alloc(buf, w_tensor, addr) == GGML_STATUS_SUCCESS;
                        if (!w_bound)
                            invalidate_cached_buffer(weight.data);
                    }
                }
            }

            ggml_tensor* mm = tsg::weight_mul_mat(ctx, w_tensor, input_binding.tensor, weight.data);
            if (mm == nullptr)
            {
                set_last_error("Failed to create the multi-rank matmul node.");
                return false;
            }

            ggml_tensor* out = keep_result_on_device ? mm : ggml_cpy(ctx, mm, rg.result_binding.tensor);
            if (out == nullptr)
            {
                set_last_error("Failed to create the multi-rank matmul output node.");
                return false;
            }
            ggml_set_output(out);

            ggml_cgraph* graph = ggml_new_graph(ctx);
            if (graph == nullptr)
            {
                set_last_error("Failed to create the multi-rank matmul graph.");
                return false;
            }
            ggml_build_forward_expand(graph, out);

            rg.owned_buffer = BufferHandle(tsg::alloc_ctx_tensors_budgeted(ctx, g_backend));
            if (rg.owned_buffer.value == nullptr)
            {
                set_last_error("Failed to allocate the multi-rank matmul backend buffer.");
                return false;
            }

            if (!zero_copy)
            {
                if (rg.packed_input.empty())
                    upload_binding(input_binding, input_desc.data, input_binding.raw_bytes);
                else
                    upload_binding(input_binding, rg.packed_input.data(), input_binding.raw_bytes);
            }
            if (!w_bound || w_needs_upload)
            {
                TensorBinding wb = { w_tensor, w_tensor, static_cast<std::size_t>(weight.raw_bytes) };
                upload_binding(wb, weight.data, wb.raw_bytes);
            }

            rg.result_node = out;
            rg.zero_copy_result = zero_copy;
            rg.result_host = result_desc.data;
            rg.result_bytes = keep_result_on_device
                ? static_cast<std::size_t>(result_desc.dim0) * result_desc.dim1 * sizeof(float)
                : rg.result_binding.raw_bytes;
            out_graph = graph;
            return true;
        }
    } // namespace
} // namespace tsg

// One column-/row-parallel linear across every rank.
//
//   results[r] = inputs[r] @ weights[r]        (r = 0 .. rankCount-1)
//
// `allReduce != 0` performs the row-parallel reduction over the per-rank
// partials before they are copied back, keeping the sum on-device when the
// backend has a collective (NCCL / P2P) and falling back to a host sum
// otherwise. With allReduce set, every rank's host buffer receives the same
// reduced result.
TSG_EXPORT int TSGgml_TensorParallelMatmul(
    tsg::TensorView2DDesc* results,
    tsg::TensorView2DDesc* inputs,
    void** weightData,
    int* weightTypes,
    int64_t* weightNe0,
    int64_t* weightNe1,
    int64_t* weightRawBytes,
    int rankCount,
    int allReduce)
{
    try
    {
        tsg::clear_last_error();
        if (results == nullptr || inputs == nullptr || weightData == nullptr ||
            rankCount <= 0 || rankCount > tsg::TSG_MAX_DEVICES)
        {
            tsg::set_last_error("Invalid arguments to the tensor-parallel matmul.");
            return 0;
        }
        if (rankCount > tsg::g_device_count.load(std::memory_order_acquire))
        {
            tsg::set_last_error("Tensor-parallel matmul rank count exceeds the initialized device count.");
            return 0;
        }

        // Reduce on-device only when the collective is available and the result
        // is a genuine row-parallel partial. Otherwise every rank downloads its
        // own slice and (for allReduce) the host sums them.
        //
        // Keeping the partial in VRAM means the download target is the packed
        // matmul output, so every rank's host buffer must be contiguous.
        bool device_reduce = allReduce != 0 && rankCount > 1 && tsg::tp_comm_ensure();
        if (device_reduce)
        {
            for (int r = 0; r < rankCount; ++r)
            {
                if (results[r].stride1 != 1 || results[r].stride0 != results[r].dim1 ||
                    results[r].dim0 != results[0].dim0 || results[r].dim1 != results[0].dim1)
                {
                    device_reduce = false;
                    break;
                }
            }
        }

        std::vector<tsg::RankGraph> graphs(static_cast<std::size_t>(rankCount));
        std::vector<ggml_cgraph*> cgraphs(static_cast<std::size_t>(rankCount), nullptr);

        // Build + submit each rank's graph without waiting: the GPUs overlap.
        for (int r = 0; r < rankCount; ++r)
        {
            tsg::ScopedRank rank(r);
            if (!tsg::ensure_backend())
                return 0;

            tsg::QuantizedWeightDesc w;
            w.data = weightData[r];
            w.ggml_type = weightTypes[r];
            w.ne0 = weightNe0[r];
            w.ne1 = weightNe1[r];
            w.raw_bytes = weightRawBytes[r];

            if (!tsg::build_rank_matmul(graphs[static_cast<std::size_t>(r)], results[r], inputs[r], w,
                    cgraphs[static_cast<std::size_t>(r)], device_reduce))
                return 0;

            if (ggml_backend_graph_compute_async(tsg::q8_f32_execution_backend(g_backend, cgraphs[static_cast<std::size_t>(r)]), cgraphs[static_cast<std::size_t>(r)]) != GGML_STATUS_SUCCESS)
            {
                tsg::set_last_error("Tensor-parallel matmul graph execution failed.");
                return 0;
            }
        }

        if (device_reduce)
        {
            // The partials are still in VRAM; reduce them there.
            std::vector<ggml_tensor*> nodes(static_cast<std::size_t>(rankCount), nullptr);
            for (int r = 0; r < rankCount; ++r)
                nodes[static_cast<std::size_t>(r)] = graphs[static_cast<std::size_t>(r)].result_node;

            if (!tsg::tp_device_allreduce(nodes.data()))
            {
                // Collective refused this payload (unsupported dtype/shape):
                // fall back to downloading and summing on the host.
                for (int r = 0; r < rankCount; ++r)
                {
                    tsg::ScopedRank rank(r);
                    tsg::sync_backend(g_backend);
                    ggml_backend_tensor_get(nodes[static_cast<std::size_t>(r)],
                        graphs[static_cast<std::size_t>(r)].result_host, 0,
                        graphs[static_cast<std::size_t>(r)].result_bytes);
                }
                std::vector<float*> bufs(static_cast<std::size_t>(rankCount));
                for (int r = 0; r < rankCount; ++r)
                    bufs[static_cast<std::size_t>(r)] = static_cast<float*>(graphs[static_cast<std::size_t>(r)].result_host);
                tsg::tp_host_allreduce_mt(bufs.data(), rankCount,
                    static_cast<std::int64_t>(graphs[0].result_bytes / sizeof(float)));
                return 1;
            }

            for (int r = 0; r < rankCount; ++r)
            {
                tsg::ScopedRank rank(r);
                tsg::sync_backend(g_backend);
                ggml_backend_tensor_get(nodes[static_cast<std::size_t>(r)],
                    graphs[static_cast<std::size_t>(r)].result_host, 0,
                    graphs[static_cast<std::size_t>(r)].result_bytes);
            }
            return 1;
        }

        // No device collective: drain each rank and copy its slice back.
        for (int r = 0; r < rankCount; ++r)
        {
            tsg::ScopedRank rank(r);
            tsg::sync_backend(g_backend);
            auto& rg = graphs[static_cast<std::size_t>(r)];
            if (!rg.zero_copy_result && rg.result_binding.storage != nullptr &&
                rg.result_host != nullptr && rg.result_bytes > 0)
            {
                ggml_backend_tensor_get(rg.result_binding.storage, rg.result_host, 0, rg.result_bytes);
            }
        }

        if (allReduce != 0 && rankCount > 1)
        {
            std::vector<float*> bufs(static_cast<std::size_t>(rankCount));
            for (int r = 0; r < rankCount; ++r)
                bufs[static_cast<std::size_t>(r)] = static_cast<float*>(graphs[static_cast<std::size_t>(r)].result_host);
            tsg::tp_host_allreduce_mt(bufs.data(), rankCount,
                static_cast<std::int64_t>(graphs[0].result_bytes / sizeof(float)));
        }
        return 1;
    }
    catch (const std::exception& ex)
    {
        tsg::set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        tsg::set_last_error("Unknown tensor-parallel matmul failure.");
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Exported entry points
// ---------------------------------------------------------------------------

// Number of GPUs the given backend type can address. C# uses this to validate
// a requested TP degree before creating a group.
TSG_EXPORT int TSGgml_GetGpuDeviceCount(int backendType)
{
    try
    {
        tsg::clear_last_error();
        if (!tsg::can_initialize_backend(backendType))
            return 0;
        // CUDA needs the registry populated before devices can be enumerated;
        // ensure_backend brings rank 0 up (and is a no-op afterwards).
        if (!tsg::ensure_backend(backendType))
            return 0;
        return tsg::gpu_device_count(backendType);
    }
    catch (const std::exception& ex)
    {
        tsg::set_last_error(ex.what());
        return 0;
    }
}

// Human-readable description of a GPU device (adapter name).
TSG_EXPORT int TSGgml_GetGpuDeviceDescription(int backendType, int deviceIndex, char* description, int descriptionSize)
{
    try
    {
        tsg::clear_last_error();
        if (description == nullptr || descriptionSize <= 0)
            return 0;
        description[0] = '\0';
        if (!tsg::ensure_backend(backendType))
            return 0;

        ggml_backend_t backend = nullptr;
        bool owned = false;
        if (deviceIndex == tsg::dev(0).device_index || (deviceIndex == 0 && tsg::dev(0).device_index < 0))
        {
            backend = tsg::dev(0).backend;
        }
        else
        {
            for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
            {
                if (tsg::dev(r).device_index == deviceIndex)
                {
                    backend = tsg::dev(r).backend;
                    break;
                }
            }
        }
        if (backend == nullptr)
        {
            backend = tsg::create_backend_instance_on_device(backendType, deviceIndex);
            owned = backend != nullptr;
        }
        if (backend == nullptr)
            return 0;

        ggml_backend_dev_t d = ggml_backend_get_device(backend);
        const char* desc = d != nullptr ? ggml_backend_dev_description(d) : nullptr;
        if (desc != nullptr)
        {
            std::snprintf(description, static_cast<std::size_t>(descriptionSize), "%s", desc);
        }
        if (owned)
            ggml_backend_free(backend);
        return desc != nullptr ? 1 : 0;
    }
    catch (const std::exception& ex)
    {
        tsg::set_last_error(ex.what());
        return 0;
    }
}

// Set a process environment variable as native code sees it.
//
// .NET's Environment.SetEnvironmentVariable only updates the CLR's own copy of
// the environment — a native getenv() never observes it. ggml reads every one
// of its tunables with getenv, and latches some of them into function-local
// statics on first use, so a managed caller that needs to influence one has to
// set it here and has to do it before the first call that reads it.
// `overwrite == 0` leaves an operator-provided value alone.
TSG_EXPORT int TSGgml_SetNativeEnvironmentVariable(const char* name, const char* value, int overwrite)
{
    if (name == nullptr || name[0] == '\0' || value == nullptr)
        return 0;
#if defined(_WIN32)
    if (overwrite == 0 && std::getenv(name) != nullptr)
        return 1;
    return _putenv_s(name, value) == 0 ? 1 : 0;
#else
    return setenv(name, value, overwrite) == 0 ? 1 : 0;
#endif
}

// Bring up `count` backends on the given physical device indices. Rank 0 reuses
// the already-initialized singleton when its device matches, so a TP run does
// not pay for a second context on the main GPU. Returns 1 on success.
static int tsg_multi_device_init(int backendType, const int* deviceIndices, int count,
                                 int concurrentRanks, bool enableCollectives)
{
    try
    {
        tsg::clear_last_error();
        if (deviceIndices == nullptr || count <= 0 || count > tsg::TSG_MAX_DEVICES)
        {
            tsg::set_last_error("Invalid tensor-parallel device list.");
            return 0;
        }

        // CUDA graph capture used to be switched OFF here for every multi-GPU
        // run, on the theory that capture is a process-wide mode which a
        // concurrent rank thread would poison ("operation failed due to a
        // previous error during capture"). That hazard no longer exists: ggml
        // captures with cudaStreamCaptureModeRelaxed, which explicitly permits
        // unrelated CUDA activity on other threads.
        //
        // Capture is worth a great deal to TP, because a tensor-parallel token
        // is dozens of small segment submissions per rank and replaying them
        // costs a fraction of re-issuing every launch. Measured on 4xA40, decode
        // tok/s, 8 requests per configuration:
        //     Qwen3.5-9B  Q8_0     tp4:  ~88 without capture, 128.5 with
        //     Qwen3.5-35B-A3B IQ4  tp2:  71.3 without capture, 104.1 with
        // The 35B case is the difference between TP being a regression against a
        // single GPU (96.0) and being a win.
        //
        // TS_GGML_TP_CUDA_GRAPHS=0 restores the old behaviour for diagnosis.
        // Setting the variable from native code (rather than managed) is
        // deliberate: .NET's Environment.SetEnvironmentVariable does not reach
        // getenv. This is only a backstop, though — ggml latches the value into
        // a function-local static on its first graph_compute, which the model
        // loader normally reaches before this runs, so the decision that takes
        // effect is made in GgmlNative's static constructor.
        const char* tp_graphs = std::getenv("TS_GGML_TP_CUDA_GRAPHS");
        if (count > 1 && tp_graphs != nullptr && tp_graphs[0] == '0')
        {
#if defined(_WIN32)
            if (std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr)
                _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
#else
            setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
#endif
        }

        // ggml-cuda's internal AllReduce converts F32 payloads to BF16 before
        // reducing them, to halve on-wire bytes. Its default threshold is 1 byte
        // — i.e. always, including the ~10 KB collectives a decode step issues,
        // where halving a PCIe transfer that small saves nothing and an 8-bit
        // mantissa on the residual stream is pure loss. Raise it so decode
        // reduces exactly in F32 and only the megabyte-scale prefill payloads
        // take the bandwidth trade, where it is worth 2.4x (measured on Gemma-4
        // E4B: 512-token prefill 1038 -> 2539 tok/s, with greedy output still
        // byte-identical to the single-GPU run). Set GGML_CUDA_AR_BF16_THRESHOLD
        // to override: 0 disables the round-trip entirely, 1 restores ggml's
        // always-on default.
        if (count > 1)
        {
#if defined(_WIN32)
            if (std::getenv("GGML_CUDA_AR_BF16_THRESHOLD") == nullptr)
                _putenv_s("GGML_CUDA_AR_BF16_THRESHOLD", "1048576");
#else
            setenv("GGML_CUDA_AR_BF16_THRESHOLD", "1048576", 0);
#endif
        }

        // Rank 0 goes through the normal init path so g_backend_type and the
        // ggml device registry are set up exactly as in a single-device run.
        if (!tsg::ensure_backend(backendType))
            return 0;

        const int available = tsg::gpu_device_count(backendType);
        for (int r = 0; r < count; ++r)
        {
            if (deviceIndices[r] < 0 || deviceIndices[r] >= available)
            {
                tsg::set_last_error("Tensor-parallel device index " + std::to_string(deviceIndices[r]) +
                    " is out of range: " + std::to_string(available) + " device(s) available.");
                return 0;
            }
            for (int q = 0; q < r; ++q)
            {
                if (deviceIndices[q] == deviceIndices[r])
                {
                    tsg::set_last_error("Duplicate device index " + std::to_string(deviceIndices[r]) +
                        " in the tensor-parallel device list.");
                    return 0;
                }
            }
        }

        // Rank 0's backend already exists. If it is not the requested device,
        // replace it — the caller picks the device order.
        {
            tsg::ScopedRank rank(0);
            // ensure_backend picked the first GPU (device 0) unless a previous
            // TensorParallelInit recorded an explicit index.
            const int current = tsg::dev(0).device_index >= 0 ? tsg::dev(0).device_index : 0;
            if (current != deviceIndices[0])
            {
                ggml_backend_t replacement = tsg::create_backend_instance_on_device(backendType, deviceIndices[0]);
                if (replacement == nullptr)
                    return 0;
                if (g_backend != nullptr)
                {
                    tsg::sync_backend(g_backend);
                    ggml_backend_free(g_backend);
                }
                g_backend = replacement;
            }
            tsg::dev(0).device_index = deviceIndices[0];
        }

        for (int r = 1; r < count; ++r)
        {
            tsg::ScopedRank rank(r);
            if (g_backend != nullptr)
            {
                // Re-init after a previous group was torn down.
                tsg::sync_backend(g_backend);
                ggml_backend_free(g_backend);
                g_backend = nullptr;
            }
            ggml_backend_t backend = tsg::create_backend_instance_on_device(backendType, deviceIndices[r]);
            if (backend == nullptr)
                return 0;
            g_backend = backend;
            tsg::dev(r).device_index = deviceIndices[r];
        }

        tsg::g_device_count.store(count, std::memory_order_release);
        // Resolve the collective backend eagerly so an NCCL/P2P init failure is
        // reported at startup rather than mid-decode. A LAYER SPLIT skips this:
        // it never reduces across devices, and bringing NCCL up anyway would
        // spend the startup time and take on the lying-P2P hang risk for a
        // collective that is never issued.
        if (count > 1 && enableCollectives)
        {
            const bool device_ar = tsg::tp_comm_ensure();
            std::fprintf(stderr,
                "[TP] GGML tensor parallelism: %d device(s), AllReduce=%s\n",
                count, device_ar ? "device (backend collective)" : "host");
            std::fflush(stderr);
        }
        else if (count > 1)
        {
            std::fprintf(stderr,
                "[GGML] multi-device (layer split): %d device(s), no collectives\n", count);
            std::fflush(stderr);
        }
        return 1;
    }
    catch (const std::exception& ex)
    {
        tsg::set_last_error(ex.what());
        return 0;
    }
}

TSG_EXPORT int TSGgml_TensorParallelInit(int backendType, const int* deviceIndices, int count, int concurrentRanks)
{
    return tsg_multi_device_init(backendType, deviceIndices, count, concurrentRanks, true);
}

// Bring up one backend per device WITHOUT a cross-device collective, for the
// layer-split path: each GPU owns a contiguous run of layers and the only thing
// that crosses a device boundary is the residual, which the caller hands over
// through host memory. Same device bring-up as TensorParallelInit in every other
// respect, so ranks, the per-device resident weight caches and the VRAM budgets
// all work identically.
TSG_EXPORT int TSGgml_MultiDeviceInit(int backendType, const int* deviceIndices, int count)
{
    return tsg_multi_device_init(backendType, deviceIndices, count, /*concurrentRanks*/ 1, false);
}

// A tensor-parallel group whose ranks are all backend instances on ONE device,
// reducing through host staging. The ranks serialize on that device, so this is
// never a performance mode: it lets sharded kernels be checked against their
// unsharded forward on a single-GPU machine (Apple Silicon has one Metal device),
// exercising the same per-rank graphs, segment schedule and reduction as a real
// group. Only the device collective (NCCL/P2P) is not exercised.
TSG_EXPORT int TSGgml_TensorParallelInitLoopback(int backendType, int count)
{
    try
    {
        tsg::clear_last_error();
        if (count < 2 || count > tsg::TSG_MAX_DEVICES)
        {
            tsg::set_last_error("Invalid loopback tensor-parallel rank count.");
            return 0;
        }
        if (!tsg::ensure_backend(backendType))
            return 0;
        tsg::dev(0).device_index = 0;
        for (int r = 1; r < count; ++r)
        {
            tsg::ScopedRank rank(r);
            if (g_backend != nullptr)
            {
                tsg::sync_backend(g_backend);
                ggml_backend_free(g_backend);
                g_backend = nullptr;
            }
            ggml_backend_t backend = tsg::create_backend_instance(backendType);
            if (backend == nullptr)
                return 0;
            g_backend = backend;
            tsg::dev(r).device_index = 0;
        }
        tsg::g_device_count.store(count, std::memory_order_release);
        std::fprintf(stderr, "[TP] loopback tensor parallelism: %d ranks on one device, AllReduce=host\n", count);
        std::fflush(stderr);
        return 1;
    }
    catch (const std::exception& ex)
    {
        tsg::set_last_error(ex.what());
        return 0;
    }
}

// 1 when the fused tensor-parallel path can run (several ranks spanning every
// initialized device). The per-layer partials reduce with the backend's device
// collective when it has one (ggml-cuda: NCCL / P2P) and through host staging
// otherwise (ggml-vulkan), so the collective is no longer a requirement.
TSG_EXPORT int TSGgml_TensorParallelFusedAvailable(int rankCount)
{
    try
    {
        return tsg::tp_fused_available(rankCount, /*distributed*/ false) ? 1 : 0;
    }
    catch (...)
    {
        return 0;
    }
}

// Availability for a DISTRIBUTED run, where one local rank per node is a
// valid configuration because the reduction spans nodes.
TSG_EXPORT int TSGgml_TensorParallelFusedAvailableDistributed(int rankCount)
{
    try
    {
        return tsg::tp_fused_available(rankCount, /*distributed*/ true) ? 1 : 0;
    }
    catch (...)
    {
        return 0;
    }
}

// Run one fused per-rank graph plan per rank: submit segment k on every rank,
// AllReduce the segment's partials on-device, advance. Plans come from a fused
// kernel called once per rank in tensor-parallel build mode (see
// TSGgml_Gemma4ModelDecode's tp_plan_out).
TSG_EXPORT int TSGgml_TensorParallelExecutePlans(void** plans, int rankCount)
{
    try
    {
        tsg::clear_last_error();
        if (plans == nullptr || rankCount <= 0 || rankCount > tsg::TSG_MAX_DEVICES)
        {
            tsg::set_last_error("Invalid tensor-parallel plan array.");
            return 0;
        }
        for (int r = 0; r < rankCount; ++r)
        {
            if (plans[r] == nullptr)
            {
                tsg::set_last_error("Null tensor-parallel plan for rank " + std::to_string(r) + ".");
                return 0;
            }
        }
        return tsg::tp_execute_plans(reinterpret_cast<tsg::TpRankPlan**>(plans), rankCount) ? 1 : 0;
    }
    catch (const std::exception& ex)
    {
        tsg::set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        tsg::set_last_error("Unknown tensor-parallel plan execution failure.");
        return 0;
    }
}

// Same schedule, but every AllReduce boundary is additionally exchanged
// across the cluster through `cross_node`. This is what lets a multi-NODE
// run keep the fused per-rank graphs instead of degrading to the per-op
// chain: the graph structure is identical, only the reduction is wider.
TSG_EXPORT int TSGgml_TensorParallelExecutePlansDistributed(
    void** plans, int rankCount, void* crossNodeCallback, void* crossNodeUser)
{
    try
    {
        tsg::clear_last_error();
        if (plans == nullptr || rankCount <= 0 || rankCount > tsg::TSG_MAX_DEVICES)
        {
            tsg::set_last_error("Invalid tensor-parallel plan array.");
            return 0;
        }
        for (int r = 0; r < rankCount; ++r)
        {
            if (plans[r] == nullptr)
            {
                tsg::set_last_error("Null tensor-parallel plan for rank " + std::to_string(r) + ".");
                return 0;
            }
        }
        return tsg::tp_execute_plans(reinterpret_cast<tsg::TpRankPlan**>(plans), rankCount,
                                     reinterpret_cast<tsg::TpDistributedAllReduce>(crossNodeCallback),
                                     crossNodeUser) ? 1 : 0;
    }
    catch (const std::exception& ex)
    {
        tsg::set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        tsg::set_last_error("Unknown distributed tensor-parallel plan execution failure.");
        return 0;
    }
}

// Select which rank subsequent ops on this thread execute on.
TSG_EXPORT int TSGgml_SetActiveDevice(int rank)
{
    if (rank < 0 || rank >= tsg::g_device_count.load(std::memory_order_acquire))
    {
        tsg::set_last_error("Active device rank out of range.");
        return 0;
    }
    tsg::g_active_rank = rank;
    return 1;
}

TSG_EXPORT int TSGgml_GetActiveDevice()
{
    return tsg::g_active_rank;
}

// Describe the CLUSTER to the kernels: how many ranks the group has in total
// and which global rank this process's rank 0 is. Set once, before any graph is
// built; a single-node run never calls it and the kernels use the local degree.
TSG_EXPORT void TSGgml_TensorParallelSetGlobalGeometry(int globalDegree, int rankOffset)
{
    tsg::g_tp_global_degree.store(globalDegree > 0 ? globalDegree : 0, std::memory_order_relaxed);
    tsg::g_tp_rank_offset.store(rankOffset > 0 ? rankOffset : 0, std::memory_order_relaxed);
}

TSG_EXPORT int TSGgml_GetTensorParallelDegree()
{
    return tsg::g_device_count.load(std::memory_order_acquire);
}

// 1 when AllReduce runs entirely on the devices (NCCL / P2P), 0 when it falls
// back to the host reduction. Surfaced so the CLI can report the transport.
//
// Creating the communicator is not evidence that it works: the backend's chain
// ends in a stub whose AllReduce always declines, so a context exists even when
// every collective will fall through to the host. Answer with a real one-element
// reduction instead, or the startup banner claims a device collective that the
// model never gets.
TSG_EXPORT int TSGgml_TensorParallelHasDeviceAllReduce()
{
    return tsg::tp_device_allreduce_usable() ? 1 : 0;
}

// Host AllReduce over `count` per-rank F32 buffers. Used by the managed TP
// group: the generic per-op path already leaves every partial in host RAM, so
// summing there avoids two PCIe crossings per collective.
TSG_EXPORT int TSGgml_TensorParallelAllReduceHost(float** buffers, int rankCount, int64_t count)
{
    try
    {
        if (buffers == nullptr || rankCount <= 1 || count <= 0)
            return 1;
        for (int r = 0; r < rankCount; ++r)
        {
            if (buffers[r] == nullptr)
            {
                tsg::set_last_error("Null buffer in tensor-parallel AllReduce.");
                return 0;
            }
        }
        tsg::tp_host_allreduce_mt(buffers, rankCount, count);
        tsg::clear_last_error();
        return 1;
    }
    catch (const std::exception& ex)
    {
        tsg::set_last_error(ex.what());
        return 0;
    }
}

// Device AllReduce over per-rank host buffers: uploads each rank's partial to
// its own GPU, reduces with the backend collective (NCCL / P2P), then downloads
// the result once per rank. Only a win when the collective is materially faster
// than the host sum — that is, on multi-GPU boxes with NVLink or working P2P and
// large payloads. Returns 0 when unavailable so the caller can use the host path.
TSG_EXPORT int TSGgml_TensorParallelAllReduceDevice(float** buffers, int rankCount, int64_t count)
{
    try
    {
        if (buffers == nullptr || rankCount <= 1 || count <= 0)
            return 1;
        if (rankCount != tsg::g_device_count.load(std::memory_order_acquire))
        {
            tsg::set_last_error("AllReduce rank count does not match the initialized device count.");
            return 0;
        }
        if (!tsg::tp_comm_ensure())
            return 0;

        std::vector<ggml_tensor*> tensors(static_cast<std::size_t>(rankCount), nullptr);
        {
            std::lock_guard<std::mutex> lock(tsg::g_ar_scratch_mutex);
            for (int r = 0; r < rankCount; ++r)
            {
                tsg::ScopedRank rank(r);
                ggml_tensor* t = tsg::ensure_ar_scratch_locked(r, count);
                if (t == nullptr)
                {
                    tsg::set_last_error("Failed to allocate the tensor-parallel AllReduce scratch buffer.");
                    return 0;
                }
                // Every rank's scratch holds a real partial that must contribute
                // to the sum. The backend collective reads
                // GGML_TENSOR_FLAG_COMPUTE to tell contributing shards from
                // inactive ones (a meta-backend concept) and ZEROES the data of
                // any shard without it on the BF16 wire path - without this
                // flag a large per-op AllReduce (>= the device threshold, i.e.
                // any prefill longer than ~128 tokens at hidden 2048) silently
                // returned all-zero sums and the per-op TP forward produced
                // garbage while every partial was correct.
                t->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_tensor_set(t, buffers[r], 0, static_cast<std::size_t>(count) * sizeof(float));
                tensors[static_cast<std::size_t>(r)] = t;
            }

            if (!tsg::tp_device_allreduce(tensors.data()))
                return 0;

            for (int r = 0; r < rankCount; ++r)
            {
                tsg::ScopedRank rank(r);
                tsg::sync_backend(g_backend);
                ggml_backend_tensor_get(tensors[static_cast<std::size_t>(r)], buffers[r], 0,
                    static_cast<std::size_t>(count) * sizeof(float));
            }
        }
        tsg::clear_last_error();
        return 1;
    }
    catch (const std::exception& ex)
    {
        tsg::set_last_error(ex.what());
        return 0;
    }
}
