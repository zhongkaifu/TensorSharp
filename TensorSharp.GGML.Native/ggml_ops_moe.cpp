// MoE FFN prefill kernel using ggml_mul_mat_id.
//
// This kernel collapses an entire MoE FFN forward pass — gate + up projections,
// SwiGLU activation, down projection, expert weighting, and per-token expert
// aggregation — into ONE GGML graph dispatch per layer per ubatch, regardless
// of how many experts are active.
//
// llama.cpp's `build_moe_ffn` is the reference: it builds the same graph using
// 2-3 `ggml_mul_mat_id` ops per layer (one for gate_up, one for down, plus an
// optional split when gate_up is fused). TensorSharp's previous MoE prefill
// path issued ~3 GGML graph submissions per active expert per layer (~3000+
// dispatches per pp2048 forward on GPT-OSS). This kernel collapses that to
// O(1) dispatches per layer.
//
// Three activation modes are supported:
//   - SwiGLU split: out = silu(gate) * up         (Qwen35, generic MoE)
//   - SwiGLU OAI : out = clamp_silu(gate, alpha)  (GPT-OSS specific variant)
//                  * (clamp(up, ±limit) + 1)
//   - GEGLU split: out = gelu(gate) * up          (Gemma 4 MoE; tanh-approx
//                                                  GELU matching llama.cpp's
//                                                  GGML_GLU_OP_GEGLU)
//   - ReLU squared: out = relu(up)^2              (Nemotron-H MoE)
//
// Two weight layouts are supported:
//   - Fused gate_up: gate weight has ne1 = 2*n_ff, up_data is null.
//   - Separate gate / up: each has ne1 = n_ff, up_data is non-null.
//
// All quantized weights are passed as a single contiguous block with a base
// pointer + total bytes. The expected layout is the per-expert quantized data
// stacked along the expert dimension (matching the original GGUF on-disk layout
// for `_exps.weight` tensors), so for mmap'd models there is zero extra copy
// cost — the C# layer simply passes the base pointer of expert 0.

#include "ggml_ops_internal.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
// ggml_graph_view: host_moe_execute_segments runs [i0, i1) slices of an
// already-built whole-model decode graph, pausing at each offloaded layer for
// the host expert matmul. Same internal API the tensor-parallel executor uses.
#include "ggml-impl.h"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

#ifdef __linux__
#include <sched.h>
#endif

using namespace tsg;

namespace tsg
{
    int available_cpu_parallelism()
    {
        const unsigned hw = std::thread::hardware_concurrency();
        int n = hw > 0 ? static_cast<int>(hw) : 1;

#ifdef __linux__
        // Affinity mask: taskset / cpuset cgroups pin the process to a subset.
        {
            cpu_set_t set;
            CPU_ZERO(&set);
            if (sched_getaffinity(0, sizeof(set), &set) == 0)
            {
                const int c = CPU_COUNT(&set);
                if (c > 0 && c < n)
                    n = c;
            }
        }

        // CFS bandwidth quota: the count the scheduler will actually let run
        // concurrently, which is what a spinning worker pool must not exceed.
        long quota = -1;
        long period = 0;
        if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r"))   // cgroup v2
        {
            char q[64] = { 0 };
            long p = 0;
            if (std::fscanf(f, "%63s %ld", q, &p) == 2 && std::strcmp(q, "max") != 0)
            {
                quota = std::atol(q);
                period = p;
            }
            std::fclose(f);
        }
        if (quota <= 0)                                            // cgroup v1
        {
            if (FILE* f = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r"))
            {
                if (std::fscanf(f, "%ld", &quota) != 1) quota = -1;
                std::fclose(f);
            }
            if (FILE* f = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r"))
            {
                if (std::fscanf(f, "%ld", &period) != 1) period = 0;
                std::fclose(f);
            }
        }
        if (quota > 0 && period > 0)
        {
            // Floor: a fractional CPU cannot carry a spinning worker, it can
            // only make every other worker wait for it at the next barrier.
            int c = static_cast<int>(quota / period);
            if (c < 1) c = 1;
            if (c < n) n = c;
        }
#endif

        return n < 1 ? 1 : n;
    }
}

namespace
{
        // ------------------------------------------------------------------
        // MoE CPU offload (llama.cpp's --n-cpu-moe / --cpu-moe equivalent)
        // ------------------------------------------------------------------
        // A dedicated, persistent ggml CPU backend used to run the routed-expert
        // FFN of layers the caller asked to keep in system RAM. It is separate
        // from g_backend (which stays on the accelerator for everything else)
        // and from the CPU backend ggml_ops_core may create when the *whole*
        // model runs on the CPU.
        //
        // Persisting it matters: ggml_backend_cpu_init allocates a thread pool,
        // and the offload path is entered once per offloaded layer per forward
        // (~32x per token for a --n-cpu-moe 32 run), so a per-call init would
        // dominate the very cost this feature exists to pay down.
        std::mutex g_moe_cpu_backend_mutex;
        ggml_backend_t g_moe_cpu_backend = nullptr;
        ggml_threadpool_t g_moe_cpu_threadpool = nullptr;

        // ONE host MoE runs at a time. The backend is a singleton with a shared
        // work buffer and a shared thread pool, so two callers computing on it
        // concurrently corrupt each other — which is exactly what a tensor-
        // parallel MoE layer does: RunPerRank drives every rank at once, and
        // with the experts offloaded each rank asks this backend for its slice.
        // (It crashed in the GPT-OSS expert-parallel path; every other caller
        // was single-threaded by luck rather than by design.) Serializing costs
        // nothing real: the pool already owns every core it is allowed to use.
        std::mutex g_moe_cpu_compute_mutex;

        // Set by TSGgml_SetHostMoeThreads (the --cpu-moe-threads flag). It
        // cannot travel through TS_CPU_MOE_THREADS alone: .NET's
        // Environment.SetEnvironmentVariable writes only the managed
        // environment on Linux, so std::getenv here never sees it and the flag
        // silently did nothing.
        std::atomic<int> g_moe_cpu_threads_override{ 0 };

        int moe_cpu_thread_count()
        {
            const int forced = g_moe_cpu_threads_override.load(std::memory_order_relaxed);
            if (forced > 0)
                return forced;
            if (const char* e = std::getenv("TS_CPU_MOE_THREADS"))
            {
                const int n = std::atoi(e);
                if (n > 0)
                    return n;
            }
            // available_cpu_parallelism, not hardware_concurrency: on a
            // cgroup-limited host the two differ by 4x and a pool sized to the
            // latter spends its life waiting at barriers (see the header).
            const int avail = available_cpu_parallelism();
            // Leave the REST OF THE PROCESS room, not just one core. The old
            // avail-1 assumed a single other busy thread; a hosted run has a GPU
            // submission thread per rank plus Kestrel and the scheduler, and
            // .NET sizes its own pool from the machine's CPU count rather than
            // the cgroup quota. Sized to the quota, the MoE pool then leaves the
            // scheduler nowhere to run those threads and every barrier waits out
            // whole timeslices.
            //
            // Measured, gemma-4-26B-A4B --tp 2 --cpu-moe on a 95-CPU quota
            // (TensorSharp.Server, 128-token generation): 71 threads 8.2 tok/s,
            // 64 threads 20.7, 56 threads 21.2 — a cliff, not a slope. The CLI,
            // which has far fewer background threads, is FLAT from 16 to 94
            // (prefill 49-62 tok/s), so the pool gains nothing above ~32 anyway
            // and capping it costs ~3% there while removing a 3x hazard.
            //
            // Small hosts keep almost everything: there the pool IS the machine
            // and the background threads are few. --cpu-moe-threads overrides.
            if (avail <= 2) return 1;
            if (avail <= 8) return avail - 1;

            // ABSOLUTE CAP, independent of how many CPUs the host has. An
            // offloaded layer at decode is ONE token wide: a few MB of expert
            // rows read from RAM and a handful of ggml ops. That is memory-bound
            // work with a barrier after every op, so past a few dozen workers
            // each extra thread adds a barrier participant and buys no
            // bandwidth — and once the pool spans both sockets of a 2-socket
            // machine it starts costing cross-socket traffic as well.
            //
            // Measured on a 2x Xeon 6952P (192 cores / 384 threads, 6 NUMA
            // nodes), gemma-4-26B-A4B --cpu-moe, ms of host matmul per 30
            // offloaded layers (lower is better):
            //
            //   threads |  8  | 16 |  32  |  64  | 96 | 192
            //   ms      | 45  | 35 | 17.6 | 17.3 | 26 | 120
            //
            // The old `avail / 2` rule picks 192 on that host and pays 7x. It
            // only looked safe because every box tested before had a cgroup
            // quota that clamped it into the flat part of the curve.
            const int n = avail / 2;
            return n > 64 ? 64 : n;
        }

        // Build (or rebuild) the pool the host MoE runs on. Owned here rather
        // than left to ggml's per-call default because of `poll`.
        //
        // ggml's default worker pool SPINS at every barrier. That is a good
        // trade when the process owns the machine, and a cliff when it does not:
        // this backend shares its cgroup quota with the accelerator submission
        // threads and, in TensorSharp.Server, with Kestrel and the scheduler. A
        // pool sized to the quota then leaves the scheduler no room to run those
        // threads, so every barrier waits out whole timeslices. Measured on the
        // 26B MoE at --tp 2 --cpu-moe: 7.0 tok/s spinning at the quota-sized
        // default against 19.8 with a pool that leaves headroom. poll = 0 makes
        // the workers block on a condvar instead, so oversubscription now costs
        // a wake-up rather than collapsing.
        void rebuild_moe_cpu_threadpool_locked()
        {
            if (g_moe_cpu_backend == nullptr)
                return;
            ggml_threadpool_params tpp = ggml_threadpool_params_default(moe_cpu_thread_count());
            tpp.poll = 0;
            ggml_threadpool_t pool = ggml_threadpool_new(&tpp);
            if (pool == nullptr)
            {
                // Fall back to ggml's own pool; the thread count still applies.
                ggml_backend_cpu_set_n_threads(g_moe_cpu_backend, moe_cpu_thread_count());
                return;
            }
            ggml_backend_cpu_set_threadpool(g_moe_cpu_backend, pool);
            ggml_backend_cpu_set_n_threads(g_moe_cpu_backend, moe_cpu_thread_count());
            if (g_moe_cpu_threadpool != nullptr)
                ggml_threadpool_free(g_moe_cpu_threadpool);
            g_moe_cpu_threadpool = pool;
        }

        ggml_backend_t moe_cpu_backend()
        {
            std::lock_guard<std::mutex> lock(g_moe_cpu_backend_mutex);
            if (g_moe_cpu_backend == nullptr)
            {
                g_moe_cpu_backend = ggml_backend_cpu_init();
                rebuild_moe_cpu_threadpool_locked();
            }
            return g_moe_cpu_backend;
        }

        // Bind a read-only weight tensor to its host data via the cacheable buffer
        // path when possible (so subsequent calls hit the cached backend buffer),
        // otherwise via the host-pointer mapped buffer (zero-copy on Metal /
        // unified memory), otherwise fall back to a pending CPU upload after the
        // backend buffer is allocated.
        struct WeightUploadInfo
        {
            ggml_tensor* tensor = nullptr;
            void* host_data = nullptr;
            std::size_t bytes = 0;
            bool needs_upload = false;
        };

        bool bind_readonly_weight_3d(
            ggml_context* ctx,
            ggml_backend_t backend,
            ggml_backend_dev_t dev,
            void* host_data,
            std::size_t bytes,
            ggml_tensor* tensor,
            std::vector<WeightUploadInfo>& uploads,
            std::vector<BufferHandle>& ephem_buffers,
            bool stream_only = false)
        {
            if (tensor == nullptr || host_data == nullptr || bytes == 0)
                return false;

            // MoE CPU offload, large-batch path: the caller wants these experts
            // computed on the accelerator but NOT resident there. Both binding
            // shortcuts below would defeat that -- the cacheable buffer keeps a
            // permanent per-tensor VRAM copy (offload would save nothing), and
            // the host-pointer buffer makes the kernel read the experts over the
            // bus element-by-element instead of as one bulk upload. Go straight
            // to the deferred upload, which lands in the REUSED graph buffer and
            // is therefore bounded by the largest single layer, not the model.
            if (stream_only)
            {
                uploads.push_back({tensor, host_data, bytes, true});
                return true;
            }

            if (dev != nullptr && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(backend, dev, tensor, host_data, bytes, buf, addr, needs_upload))
                {
                    if (ggml_backend_tensor_alloc(buf, tensor, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload)
                        {
                            uploads.push_back({tensor, host_data, bytes, true});
                        }
                        return true;
                    }
                    invalidate_cached_buffer(host_data);
                }
            }

            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(backend, dev, host_data, bytes, false, buf))
                {
                    ephem_buffers.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, tensor, host_data) == GGML_STATUS_SUCCESS)
                    {
                        return true;
                    }
                }
            }

            // Fall back to deferred upload after backend buffer allocation.
            uploads.push_back({tensor, host_data, bytes, true});
            return true;
        }

        // Build the chain of adds that aggregates the per-expert outputs along
        // the n_used dimension. Equivalent to:
        //   for u = 0..n_used-1: out += weighted[:, u, :]
        // but expressed as a chain of ggml_add over views so the scheduler can
        // see it as a single graph.
        ggml_tensor* build_expert_aggregation(
            ggml_context* ctx,
            ggml_tensor* weighted,
            int hidden_dim,
            int n_used,
            int seq_len)
        {
            std::vector<ggml_tensor*> views(n_used);
            for (int u = 0; u < n_used; ++u)
            {
                views[u] = ggml_view_2d(
                    ctx,
                    weighted,
                    static_cast<std::int64_t>(hidden_dim),
                    static_cast<std::int64_t>(seq_len),
                    weighted->nb[2],
                    static_cast<std::size_t>(u) * weighted->nb[1]);
                if (views[u] == nullptr)
                    return nullptr;
            }
            ggml_tensor* acc = views[0];
            for (int u = 1; u < n_used; ++u)
            {
                acc = ggml_add(ctx, acc, views[u]);
                if (acc == nullptr)
                    return nullptr;
            }
            return acc;
        }

        // Activation modes for the fused MoE prefill kernel.
        enum MoEActivation : int
        {
            MOE_ACT_SWIGLU_SPLIT = 0, // silu(gate) * up
            MOE_ACT_SWIGLU_OAI   = 1, // gpt-oss clamped variant
            MOE_ACT_GEGLU_SPLIT  = 2, // gelu(gate) * up   (Gemma 4 MoE)
            MOE_ACT_RELU_SQUARED = 3, // relu(up)^2        (Nemotron-H MoE)
        };

    // Parameter bag for the unified MoE FFN graph builder. The two public
    // entry points (the original `MoEFFNPrefill` writing to `hidden_out` and
    // the Gemma 4 residual-fused variant writing through
    // `residual_in_out += rms_norm(moe_out, post_norm_w)`) only differ in the
    // graph node sequence after the per-expert aggregation, so the body is
    // shared and parameterised through this struct.
    struct MoEFFNGraphParams
    {
        // Common inputs
        float* hidden_in = nullptr;
        int seq_len = 0;
        int hidden_dim = 0;
        int n_ff = 0;
        int num_experts = 0;
        int n_used = 0;
        const std::int32_t* selected_experts = nullptr;
        const float* routing_weights = nullptr;

        void* gate_data = nullptr;
        int gate_type = 0;
        std::int64_t gate_ne0 = 0;
        std::int64_t gate_ne1 = 0;
        std::int64_t gate_total_bytes = 0;

        void* up_data = nullptr;
        int up_type = 0;
        std::int64_t up_ne0 = 0;
        std::int64_t up_ne1 = 0;
        std::int64_t up_total_bytes = 0;

        void* down_data = nullptr;
        int down_type = 0;
        std::int64_t down_ne0 = 0;
        std::int64_t down_ne1 = 0;
        std::int64_t down_total_bytes = 0;

        const float* gate_bias = nullptr;
        const float* up_bias = nullptr;
        const float* down_bias = nullptr;

        int activation_type = 0;
        float oai_alpha = 1.702f;
        float oai_limit = 7.0f;

        // Output mode A: write moe_out to a dedicated buffer (no residual fold).
        // Used by the standalone TSGgml_MoEFFNPrefillSwiGLUQuantF32.
        float* hidden_out = nullptr;

        // Output mode B (Gemma 4 residual variant): the kernel computes
        //   residual_in_out += rms_norm(moe_out, eps) * post_norm_w
        // entirely on-device. Both pointers must be non-null and post_norm_w
        // must point to a hidden_dim-element F32 weight vector. eps is the
        // RMSNorm epsilon (ModelConfig.Eps in C#).
        float* residual_in_out = nullptr;
        const float* post_norm_w = nullptr;
        float post_norm_eps = 1e-6f;

        // MoE CPU offload: run this layer's expert matmuls on the host ggml CPU
        // backend, reading the stacked expert weights zero-copy out of the GGUF
        // mmap instead of uploading them to the accelerator. Set per layer by
        // the caller from MoeCpuOffloadConfig.IsLayerOnCpu(layer).
        bool run_on_cpu = false;

        // Whether run_on_cpu may be satisfied by streaming the experts to the
        // accelerator for one graph instead of computing them on the host (see
        // moe_offload_stream_batch_threshold). Cleared by tensor parallelism,
        // where the offloaded layer is deliberately evaluated once on the host
        // over the UNSHARDED expert stack and the ranks differ only in where
        // the single result lands (HostMoeSegment::tp_reduced).
        bool allow_device_stream = true;
    };

    // MoE CPU offload: the batch size at or above which an offloaded layer is
    // computed on the accelerator with its experts streamed in for that one
    // graph, rather than on the host.
    //
    // Offload exists to keep the experts OUT of VRAM, and at decode (one token)
    // the host is the right answer: the matmul is tiny and streaming hundreds of
    // MB per layer per token would be absurd. Prefill inverts that -- the upload
    // is a FIXED cost per layer that the whole batch amortizes, and the
    // arithmetic is what the GPU is for. llama.cpp makes the same trade in its
    // scheduler (ggml_backend_sched_backend_id_from_cur ->
    // ggml_backend_cuda_device_offload_op), which is the entire reason its
    // prefill with -ncmoe stays 4-8x ahead of a host-only seam.
    //
    // The crossover is where (expert bytes / PCIe bandwidth) meets
    // (batch * per-token host matmul). Measured on gemma-4-26B-A4B --cpu-moe,
    // RTX 3080 Laptop + i7-11800H (12.1 GiB of experts, ~8 GB/s pageable H2D,
    // best of 3, prefill ms, lower is better):
    //
    //   batch |  32  |  64  | 128  | 256
    //   host  |  580 | 1081 | 2309 | 4266
    //   strm  | 1857 | 1234 | 1254 | 1329
    //
    // i.e. streaming LOSES 3.2x at 32 and 1.1x at 64, then wins 1.8x at 128 and
    // 3.2x at 256. llama.cpp's own constant here is 32, which on this box is on
    // the wrong side of its own crossover; 128 sits just past it with margin and
    // still catches every real prefill (chunked at ubatch 256/512) while leaving
    // small block-decode batches -- DiffusionGemma's denoising blocks -- on the
    // host where they belong.
    //
    // 0 disables the device path (TS_HOST_MOE_DEVICE_MIN_BATCH=0), restoring
    // host-only offload.
    int moe_offload_stream_batch_threshold()
    {
        static const int s_threshold = []() {
            if (const char* e = std::getenv("TS_HOST_MOE_DEVICE_MIN_BATCH"))
            {
                const int n = std::atoi(e);
                if (n >= 0)
                    return n;
            }
            return 128;
        }();
        return s_threshold;
    }

    // ------------------------------------------------------------------
    // Streamed-expert upload (MoE CPU offload, prefill-sized batches)
    // ------------------------------------------------------------------
    // An offloaded layer above the batch threshold is computed on the
    // accelerator with its experts pushed over the bus for that one graph. Two
    // things decide how long that push takes, and both are handled here:
    //
    //  1. WHICH BYTES. Only the experts this batch actually routes to are
    //     needed. llama.cpp's scheduler does the same (ggml-backend.cpp builds
    //     a used-expert bitset from the ids tensor and copies consecutive runs);
    //     the saving is large at the small batches speculative verification and
    //     lightly-loaded serving produce, and nil at 512 tokens where every
    //     expert is drawn at least once. A trailing pad matches llama.cpp's:
    //     MMQ reads a tile past the last row it needs, and whatever the reused
    //     graph buffer happened to hold there could decode to NaN.
    //
    //  2. HOW FAST. See host_pin_range: page-locking the mmap turns a 9.3 GB/s
    //     staged copy into a 55.6 GB/s DMA on PCIe 5.0.
    //
    // The copies are issued with ggml_backend_tensor_set_async and land on the
    // backend's own stream, so they are ordered before the compute that follows
    // without a host round trip per tensor (the old code synchronized three
    // times per offloaded layer).
    struct ExpertRuns
    {
        // [first, last] inclusive expert index runs, in ascending order.
        std::vector<std::pair<int, int>> runs;
        int used = 0;
    };

    void build_expert_runs(const std::int32_t* ids, std::size_t count, int num_experts, ExpertRuns& out)
    {
        out.runs.clear();
        out.used = 0;
        if (ids == nullptr || count == 0 || num_experts <= 0)
            return;

        static thread_local std::vector<std::uint8_t> s_seen;
        s_seen.assign(static_cast<std::size_t>(num_experts), 0);
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::int32_t e = ids[i];
            if (e >= 0 && e < num_experts && !s_seen[static_cast<std::size_t>(e)])
            {
                s_seen[static_cast<std::size_t>(e)] = 1;
                ++out.used;
            }
        }
        int run_start = -1;
        for (int e = 0; e < num_experts; ++e)
        {
            if (s_seen[static_cast<std::size_t>(e)])
            {
                if (run_start < 0)
                    run_start = e;
            }
            else if (run_start >= 0)
            {
                out.runs.emplace_back(run_start, e - 1);
                run_start = -1;
            }
        }
        if (run_start >= 0)
            out.runs.emplace_back(run_start, num_experts - 1);
    }

    // Push a stacked expert weight to the device, restricted to `runs` when the
    // stacked layout is known to be contiguous per expert. Returns the bytes
    // actually transferred.
    std::size_t stream_expert_weight(
        ggml_backend_t backend,
        ggml_tensor* tensor,
        const void* host_data,
        std::size_t total_bytes,
        int num_experts,
        const ExpertRuns& runs)
    {
        if (tensor == nullptr || host_data == nullptr || total_bytes == 0)
            return 0;

        // Page-locking is a CUDA driver call: only worth doing — and only safe
        // to do — when the accelerator this streams to IS CUDA. On a Vulkan or
        // Metal run cudaHostRegister would spin up a CUDA primary context (and
        // its VRAM) for a device nothing is using.
        const bool pinned = (g_backend_type == BACKEND_TYPE_CUDA) &&
                            host_pin_range(host_data, total_bytes);

        // TS_HOST_MOE_TIMING=2: per-weight transfer rate. Needs a synchronize
        // after each tensor, so it changes what it measures — only for finding
        // out whether the pin actually took.
        static const int s_probe = []() {
            const char* e = std::getenv("TS_HOST_MOE_TIMING");
            return (e != nullptr && e[0] == '2') ? 1 : 0;
        }();
        std::chrono::steady_clock::time_point t_probe;
        if (s_probe)
            t_probe = std::chrono::steady_clock::now();

        // TS_HOST_MOE_EXPERT_FILTER=0 sends the whole stack every time (the
        // A/B for "is a partially-populated expert buffer to blame").
        static const bool s_filter = []() {
            const char* e = std::getenv("TS_HOST_MOE_EXPERT_FILTER");
            return !(e != nullptr && e[0] == '0');
        }();

        const std::size_t stride = num_experts > 0 ? tensor->nb[2] : 0;
        const bool per_expert_ok =
            s_filter &&
            num_experts > 0 && stride > 0 &&
            stride * static_cast<std::size_t>(num_experts) == total_bytes &&
            !runs.runs.empty() &&
            runs.used < num_experts;

        std::size_t sent = 0;
        if (!per_expert_ok)
        {
            ggml_backend_tensor_set_async(backend, tensor, host_data, 0, total_bytes);
            sent = total_bytes;
        }
        else
        {
            const std::size_t pad = stride < 512 ? stride : 512;
            for (const auto& run : runs.runs)
            {
                const std::size_t offset = static_cast<std::size_t>(run.first) * stride;
                std::size_t len = static_cast<std::size_t>(run.second - run.first + 1) * stride;
                if (run.second < num_experts - 1)
                    len += pad;                   // MMQ tile overrun guard
                if (offset + len > total_bytes)
                    len = total_bytes - offset;
                ggml_backend_tensor_set_async(backend, tensor,
                    static_cast<const std::uint8_t*>(host_data) + offset, offset, len);
                sent += len;
            }
        }

        if (s_probe)
        {
            tsg::sync_backend(backend);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_probe).count();
            static thread_local int s_n = 0;
            if (s_n++ < 12)
            {
                std::fprintf(stderr, "[HOSTMOE-COPY] %s pinned=%d copies=%d %.1f MiB in %.2f ms = %.1f GB/s\n",
                    ggml_get_name(tensor), pinned ? 1 : 0,
                    per_expert_ok ? (int) runs.runs.size() : 1,
                    sent / 1048576.0, ms, sent / 1e9 / (ms / 1000.0));
                std::fflush(stderr);
            }
        }
        return sent;
    }

    // Execute an already-built MoE FFN graph on the host CPU backend.
    //
    // Every buffer this path touches is host memory the caller already owns, so
    // each large tensor is wrapped zero-copy with ggml_backend_cpu_buffer_from_ptr:
    //   * gate / up / down: the stacked per-expert GGUF blocks, still mmap'd.
    //     Nothing is copied and nothing is uploaded, which is precisely the VRAM
    //     that offload is buying back.
    //   * hidden_in / hidden_out: the caller's F32 activation tensors.
    // Only the tiny ids / routing-weight / bias tensors and the graph's
    // intermediates get a real allocation.
    //
    // Note this deliberately does NOT go through try_get_cacheable_tensor_buffer:
    // that cache is keyed on the host pointer and would hand back an accelerator
    // buffer previously created for the same weight, silently putting the bytes
    // back in VRAM.
    struct MoECpuGraphBinding
    {
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* hidden_out_t = nullptr;
        ggml_tensor* post_norm_w_t = nullptr;
        ggml_tensor* gate_w = nullptr;
        ggml_tensor* up_w = nullptr;
        ggml_tensor* down_w = nullptr;
        ggml_tensor* ids_t = nullptr;
        ggml_tensor* weights_t = nullptr;
        ggml_tensor* gate_bias_t = nullptr;
        ggml_tensor* up_bias_t = nullptr;
        ggml_tensor* down_bias_t = nullptr;
        std::int64_t gate_bias_dim = 0;
        std::size_t hidden_bytes = 0;
        float* out_buffer_ptr = nullptr;
    };

    int moe_ffn_execute_on_cpu(
        ggml_context* ctx,
        ggml_cgraph* graph,
        const MoEFFNGraphParams& p,
        const MoECpuGraphBinding& b)
    {
        const auto t_enter = std::chrono::steady_clock::now();
        ggml_backend_t cpu = moe_cpu_backend();
        if (cpu == nullptr)
        {
            set_last_error("MoE CPU offload: failed to initialise the host ggml CPU backend.");
            return 0;
        }
        // Held for the whole body: the scratch allocation, the host-pointer
        // wraps and the compute all run against the one shared backend.
        std::lock_guard<std::mutex> compute_lock(g_moe_cpu_compute_mutex);

        // The accelerator may still have writes in flight targeting hidden_in
        // (the previous layer's output) and the routing arrays. The host is
        // about to read them directly, so drain first.
        host_read_barrier();

        std::vector<BufferHandle> buffers;
        auto wrap_host = [&](ggml_tensor* t, void* data, std::size_t bytes) -> bool
        {
            if (t == nullptr || data == nullptr || bytes == 0)
                return false;

            // ggml_backend_cpu_buffer_from_ptr ASSERTS TENSOR_ALIGNMENT (32) on
            // the pointer -- it aborts the process, it does not return null. A
            // stacked expert tensor inside a memory-mapped GGUF lands wherever
            // its row size puts it, and quantized rows are not multiples of 32
            // bytes (an IQ4_XS row of 2816 elements is 11 blocks = 1496 bytes),
            // so gemma-4-26B-A4B-IQ4_XS aborted here on the first --n-cpu-moe
            // decode. Wrap an aligned window that CONTAINS the tensor and give
            // the tensor its true address: only the buffer constructor is
            // picky, ggml's CPU kernels load unaligned. Aligning DOWN cannot
            // leave the mapping -- the mmap base is page-aligned, so a pointer
            // within 31 bytes of it is already aligned -- and the lead bytes
            // are never read.
            const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(data);
            const std::uintptr_t base = raw & ~static_cast<std::uintptr_t>(TENSOR_ALIGNMENT - 1);
            const std::size_t lead = static_cast<std::size_t>(raw - base);

            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(
                reinterpret_cast<void*>(base), bytes + lead);
            if (buf == nullptr)
                return false;
            buffers.emplace_back(buf);
            return ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS;
        };

        if (!wrap_host(b.hidden_t, p.hidden_in, b.hidden_bytes) ||
            !wrap_host(b.hidden_out_t, b.out_buffer_ptr, b.hidden_bytes) ||
            !wrap_host(b.gate_w, p.gate_data, static_cast<std::size_t>(p.gate_total_bytes)) ||
            !wrap_host(b.down_w, p.down_data, static_cast<std::size_t>(p.down_total_bytes)))
        {
            set_last_error("MoE CPU offload: failed to wrap a host buffer for the expert weights.");
            return 0;
        }
        if (b.up_w != nullptr &&
            !wrap_host(b.up_w, p.up_data, static_cast<std::size_t>(p.up_total_bytes)))
        {
            set_last_error("MoE CPU offload: failed to wrap the up-projection host buffer.");
            return 0;
        }

        // Everything still unbound (ids, routing weights, biases, post-norm
        // weight, and all graph intermediates) is small; allocate it in one
        // CPU-backend buffer freed when this call returns.
        BufferHandle scratch(ggml_backend_alloc_ctx_tensors(ctx, cpu));
        if (scratch.value == nullptr)
        {
            set_last_error("MoE CPU offload: failed to allocate the host scratch buffer.");
            return 0;
        }

        ggml_backend_tensor_set(b.ids_t, p.selected_experts, 0,
            static_cast<std::size_t>(p.seq_len) *
            static_cast<std::size_t>(p.n_used) * sizeof(std::int32_t));
        ggml_backend_tensor_set(b.weights_t, p.routing_weights, 0,
            static_cast<std::size_t>(p.seq_len) *
            static_cast<std::size_t>(p.n_used) * sizeof(float));

        if (b.post_norm_w_t != nullptr)
        {
            ggml_backend_tensor_set(b.post_norm_w_t, p.post_norm_w, 0,
                static_cast<std::size_t>(p.hidden_dim) * sizeof(float));
        }
        if (b.gate_bias_t != nullptr)
        {
            ggml_backend_tensor_set(b.gate_bias_t, p.gate_bias, 0,
                static_cast<std::size_t>(b.gate_bias_dim) *
                static_cast<std::size_t>(p.num_experts) * sizeof(float));
        }
        if (b.up_bias_t != nullptr)
        {
            ggml_backend_tensor_set(b.up_bias_t, p.up_bias, 0,
                static_cast<std::size_t>(p.n_ff) *
                static_cast<std::size_t>(p.num_experts) * sizeof(float));
        }
        if (b.down_bias_t != nullptr)
        {
            ggml_backend_tensor_set(b.down_bias_t, p.down_bias, 0,
                static_cast<std::size_t>(p.hidden_dim) *
                static_cast<std::size_t>(p.num_experts) * sizeof(float));
        }

        // TS_HOST_MOE_TIMING=1: where an offloaded layer's wall clock goes.
        // Split out because "the host matmul is slow" and "the per-call scratch
        // allocation is slow" want completely different fixes.
        static const bool s_timing = []() {
            const char* e = std::getenv("TS_HOST_MOE_TIMING");
            return e != nullptr && e[0] == '1';
        }();
        const auto t_compute = std::chrono::steady_clock::now();
        const ggml_status status = tsg::compute_graph(cpu, graph);
        if (s_timing)
        {
            const auto t_end = std::chrono::steady_clock::now();
            static thread_local double acc_setup = 0.0, acc_compute = 0.0;
            static thread_local int acc_calls = 0;
            acc_setup += std::chrono::duration<double, std::milli>(t_compute - t_enter).count();
            acc_compute += std::chrono::duration<double, std::milli>(t_end - t_compute).count();
            if (++acc_calls % 30 == 0)
            {
                std::fprintf(stderr, "[HOSTMOE-TIME] seq=%d calls=%d setup=%.1f ms compute=%.1f ms\n",
                    p.seq_len, acc_calls, acc_setup, acc_compute);
                std::fflush(stderr);
                acc_setup = acc_compute = 0.0;
            }
        }
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("MoE CPU offload: host graph compute failed.");
            return 0;
        }

        // hidden_out_t was wrapped around the caller's buffer, so the result is
        // already where the caller expects it - no download, no barrier.
        clear_last_error();
        return 1;
    }

    int moe_ffn_prefill_graph_impl(const MoEFFNGraphParams& p)
    {
        float* const hidden_in = p.hidden_in;
        const int seq_len = p.seq_len;
        const int hidden_dim = p.hidden_dim;
        const int n_ff = p.n_ff;
        const int num_experts = p.num_experts;
        const int n_used = p.n_used;
        const std::int32_t* const selected_experts = p.selected_experts;
        const float* const routing_weights = p.routing_weights;
        void* const gate_data = p.gate_data;
        const int gate_type = p.gate_type;
        const std::int64_t gate_ne0 = p.gate_ne0;
        const std::int64_t gate_ne1 = p.gate_ne1;
        const std::int64_t gate_total_bytes = p.gate_total_bytes;
        void* const up_data = p.up_data;
        const int up_type = p.up_type;
        const std::int64_t up_ne0 = p.up_ne0;
        const std::int64_t up_ne1 = p.up_ne1;
        const std::int64_t up_total_bytes = p.up_total_bytes;
        void* const down_data = p.down_data;
        const int down_type = p.down_type;
        const std::int64_t down_ne0 = p.down_ne0;
        const std::int64_t down_ne1 = p.down_ne1;
        const std::int64_t down_total_bytes = p.down_total_bytes;
        const float* const gate_bias = p.gate_bias;
        const float* const up_bias = p.up_bias;
        const float* const down_bias = p.down_bias;
        const int activation_type = p.activation_type;
        const float oai_alpha = p.oai_alpha;
        const float oai_limit = p.oai_limit;
        float* const hidden_out = p.hidden_out;
        float* const residual_in_out = p.residual_in_out;
        const float* const post_norm_w = p.post_norm_w;
        const float post_norm_eps = p.post_norm_eps;
        // Output mode B (residual fold) is selected when both the residual
        // buffer and the post-norm weight are supplied. Mutually exclusive
        // with mode A.
        const bool residual_mode = (residual_in_out != nullptr) && (post_norm_w != nullptr);
        const bool standalone_mode = (hidden_out != nullptr) && !residual_mode;
        if (!ensure_backend()) return 0;

        // --- Argument validation ---
        if (!residual_mode && !standalone_mode)
        {
            set_last_error("MoE prefill: caller must supply either hidden_out (standalone) or both residual_in_out and post_norm_w (residual mode).");
            return 0;
        }
        if (residual_mode && hidden_out != nullptr)
        {
            set_last_error("MoE prefill: residual_in_out and hidden_out are mutually exclusive output modes.");
            return 0;
        }
        if (hidden_in == nullptr || selected_experts == nullptr || routing_weights == nullptr)
        {
            set_last_error("MoE prefill: null host pointer for input/ids/weights.");
            return 0;
        }
        if (seq_len <= 0 || hidden_dim <= 0 || n_ff <= 0 || num_experts <= 0 || n_used <= 0)
        {
            set_last_error("MoE prefill: invalid shape parameter (must all be > 0).");
            return 0;
        }
        if (n_used > num_experts)
        {
            set_last_error("MoE prefill: n_used cannot exceed num_experts.");
            return 0;
        }
        if (residual_mode && !std::isfinite(post_norm_eps))
        {
            set_last_error("MoE prefill: post_norm_eps must be a finite positive value.");
            return 0;
        }

        const bool single_projection_activation = (activation_type == MOE_ACT_RELU_SQUARED);
        const bool fused_gate_up = (up_data == nullptr) && !single_projection_activation;
        const std::int64_t expected_gate_ne1 = fused_gate_up
            ? static_cast<std::int64_t>(2) * n_ff
            : static_cast<std::int64_t>(n_ff);

        if (gate_data == nullptr ||
            gate_ne0 != static_cast<std::int64_t>(hidden_dim) ||
            gate_ne1 != expected_gate_ne1 ||
            gate_total_bytes <= 0)
        {
            set_last_error("MoE prefill: gate weight pointer/shape mismatch.");
            return 0;
        }
        if (single_projection_activation && up_data != nullptr)
        {
            set_last_error("MoE prefill: ReLU-squared activation expects up_data to be null; pass the up projection as gate_data.");
            return 0;
        }
        if (!fused_gate_up && !single_projection_activation &&
            (up_ne0 != static_cast<std::int64_t>(hidden_dim) ||
             up_ne1 != static_cast<std::int64_t>(n_ff) ||
             up_total_bytes <= 0))
        {
            set_last_error("MoE prefill: up weight pointer/shape mismatch.");
            return 0;
        }
        if (down_data == nullptr ||
            down_ne0 != static_cast<std::int64_t>(n_ff) ||
            down_ne1 != static_cast<std::int64_t>(hidden_dim) ||
            down_total_bytes <= 0)
        {
            set_last_error("MoE prefill: down weight pointer/shape mismatch.");
            return 0;
        }

        const std::size_t hidden_bytes =
            static_cast<std::size_t>(seq_len) *
            static_cast<std::size_t>(hidden_dim) * sizeof(float);

        // --- Context allocation ---
        // Conservative size to fit graph nodes + ephemeral tensor headers for
        // up to ~16 active experts in the aggregation chain.
        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("MoE prefill: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        // --- Tensor allocation ---
        // Hidden state tensors are 2D [hidden_dim, seq_len] (column-major as
        // ggml expects; ne0 is the innermost / leading dim and seq_len is the
        // outer dim, matching the row-major C# layout
        //   hidden[s * hidden_dim + d]
        // which equals ggml's hidden_t[d, s]).
        ggml_tensor* hidden_t = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, hidden_dim, seq_len);
        // In standalone mode this holds the MoE output. In residual mode the
        // residual buffer (passed in by the caller, holds the dense FFN output
        // we add the normed MoE result to) plays the same structural role.
        ggml_tensor* hidden_out_t = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, hidden_dim, seq_len);

        // Residual-mode-only weight: F32 [hidden_dim] post_ffw_norm_2.weight,
        // multiplied with rms_norm(moe_out) before the residual add.
        ggml_tensor* post_norm_w_t = nullptr;
        if (residual_mode)
        {
            post_norm_w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);
        }

        ggml_tensor* gate_w = ggml_new_tensor_3d(
            ctx, static_cast<ggml_type>(gate_type),
            gate_ne0, gate_ne1, num_experts);
        ggml_tensor* up_w = nullptr;
        if (!fused_gate_up && !single_projection_activation)
        {
            up_w = ggml_new_tensor_3d(
                ctx, static_cast<ggml_type>(up_type),
                up_ne0, up_ne1, num_experts);
        }
        ggml_tensor* down_w = ggml_new_tensor_3d(
            ctx, static_cast<ggml_type>(down_type),
            down_ne0, down_ne1, num_experts);

        // ids: i32 [n_used, seq_len] — innermost dim n_used so memory layout
        // matches selected_experts[s * n_used + u].
        ggml_tensor* ids_t = ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, n_used, seq_len);

        // routing weights: [1, n_used, seq_len] — broadcasting scalar per
        // (token, used-expert) slot. Layout matches routing_weights[s * n_used + u].
        ggml_tensor* weights_t = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, 1, n_used, seq_len);

        // Optional per-expert biases. ggml_add_id consumes a 2D [bias_dim, num_experts]
        // f32 tensor and adds bias[:, ids[u, t]] to result[:, u, t].
        ggml_tensor* gate_bias_t = nullptr;
        ggml_tensor* up_bias_t = nullptr;
        ggml_tensor* down_bias_t = nullptr;
        const std::int64_t gate_bias_dim = fused_gate_up
            ? static_cast<std::int64_t>(2) * n_ff
            : static_cast<std::int64_t>(n_ff);
        if (gate_bias != nullptr)
        {
            gate_bias_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, gate_bias_dim, num_experts);
        }
        if (!fused_gate_up && !single_projection_activation && up_bias != nullptr)
        {
            up_bias_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, num_experts);
        }
        if (down_bias != nullptr)
        {
            down_bias_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, num_experts);
        }

        if (hidden_t == nullptr || hidden_out_t == nullptr ||
            gate_w == nullptr || down_w == nullptr ||
            ids_t == nullptr || weights_t == nullptr ||
            (!fused_gate_up && !single_projection_activation && up_w == nullptr))
        {
            set_last_error("MoE prefill: failed to create ggml tensors.");
            return 0;
        }

        // Keep every small host-uploaded parameter out of the allocator's free
        // list. A graph allocator frees a leaf after its last consumer node and
        // may place a later activation on top of it, which is safe only while
        // every node runs on its own. Backends fuse nodes: ggml-cuda runs
        // {mul_mat_id, add_id, mul_mat_id, add_id, swiglu} as ONE MMVQ kernel for
        // a small token batch, reading the gate/up biases while it writes the
        // activation, and its overlap check skips leafs because llama.cpp's biases
        // are weights, never compute memory. Unpinned, the activation landed on the
        // gate bias and the kernel overwrote the bias it was reading: wrong, run-to-
        // run different expert outputs (gpt-oss batched decode of 2-7 sequences).
        // An output flag is the allocator's "never free"; these tensors are a few
        // hundred KB at most. The hidden input is not pinned: it can be tokens x
        // hidden large, and every kernel that reads it (the expert matmuls)
        // quantizes or gathers it before writing anything.
        for (ggml_tensor* uploaded : { ids_t, weights_t, gate_bias_t, up_bias_t, down_bias_t, post_norm_w_t })
        {
            if (uploaded != nullptr)
            {
                ggml_set_input(uploaded);
                ggml_set_output(uploaded);
            }
        }

        // --- Build graph ---
        // Reshape input to [hidden_dim, 1, seq_len] so mul_mat_id can broadcast
        // it across the n_used expert slots per token.
        ggml_tensor* cur_3d = ggml_reshape_3d(ctx, hidden_t, hidden_dim, 1, seq_len);
        if (cur_3d == nullptr)
        {
            set_last_error("MoE prefill: failed to reshape input.");
            return 0;
        }

        ggml_tensor* gate_proj = nullptr;
        ggml_tensor* up_proj = nullptr;

        if (single_projection_activation)
        {
            gate_proj = ggml_mul_mat_id(ctx, gate_w, cur_3d, ids_t);
            if (gate_proj == nullptr)
            {
                set_last_error("MoE prefill: ggml_mul_mat_id(up) failed.");
                return 0;
            }
            if (gate_bias_t != nullptr)
            {
                gate_proj = ggml_add_id(ctx, gate_proj, gate_bias_t, ids_t);
                if (gate_proj == nullptr)
                {
                    set_last_error("MoE prefill: ggml_add_id(up_bias) failed.");
                    return 0;
                }
            }
        }
        else if (fused_gate_up)
        {
            // gate_w shape: [hidden_dim, 2*n_ff, num_experts]
            // result shape: [2*n_ff, n_used, seq_len]
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, gate_w, cur_3d, ids_t);
            if (gate_up == nullptr)
            {
                set_last_error("MoE prefill: ggml_mul_mat_id(gate_up) failed.");
                return 0;
            }
            if (gate_bias_t != nullptr)
            {
                gate_up = ggml_add_id(ctx, gate_up, gate_bias_t, ids_t);
                if (gate_up == nullptr)
                {
                    set_last_error("MoE prefill: ggml_add_id(gate_up_bias) failed.");
                    return 0;
                }
            }
            // Split into two views over the leading dim.
            // gate = view at offset 0, up = view at offset n_ff * nb[0].
            gate_proj = ggml_view_3d(ctx, gate_up,
                static_cast<std::int64_t>(n_ff),
                gate_up->ne[1],
                gate_up->ne[2],
                gate_up->nb[1],
                gate_up->nb[2],
                0);
            up_proj = ggml_view_3d(ctx, gate_up,
                static_cast<std::int64_t>(n_ff),
                gate_up->ne[1],
                gate_up->ne[2],
                gate_up->nb[1],
                gate_up->nb[2],
                static_cast<std::size_t>(n_ff) * gate_up->nb[0]);
        }
        else
        {
            gate_proj = ggml_mul_mat_id(ctx, gate_w, cur_3d, ids_t);
            up_proj = ggml_mul_mat_id(ctx, up_w, cur_3d, ids_t);
            if (gate_proj == nullptr || up_proj == nullptr)
            {
                set_last_error("MoE prefill: failed to build gate/up projections.");
                return 0;
            }
            if (gate_bias_t != nullptr)
            {
                gate_proj = ggml_add_id(ctx, gate_proj, gate_bias_t, ids_t);
            }
            if (up_bias_t != nullptr)
            {
                up_proj = ggml_add_id(ctx, up_proj, up_bias_t, ids_t);
            }
        }

        if (gate_proj == nullptr || (!single_projection_activation && up_proj == nullptr))
        {
            set_last_error("MoE prefill: failed to build gate/up projections.");
            return 0;
        }

        ggml_tensor* activated = nullptr;
        switch (activation_type)
        {
            case MOE_ACT_SWIGLU_SPLIT:
                activated = ggml_swiglu_split(ctx, gate_proj, up_proj);
                break;
            case MOE_ACT_SWIGLU_OAI:
                activated = ggml_swiglu_oai(ctx, gate_proj, up_proj, oai_alpha, oai_limit);
                break;
            case MOE_ACT_GEGLU_SPLIT:
                activated = ggml_geglu_split(ctx, gate_proj, up_proj);
                break;
            case MOE_ACT_RELU_SQUARED:
            {
                ggml_tensor* relu_out = ggml_relu(ctx, gate_proj);
                if (relu_out == nullptr)
                {
                    set_last_error("MoE prefill: ReLU activation node creation failed.");
                    return 0;
                }
                activated = ggml_sqr(ctx, relu_out);
                break;
            }
            default:
                set_last_error("MoE prefill: unknown activation_type.");
                return 0;
        }

        if (activated == nullptr)
        {
            set_last_error("MoE prefill: activation node creation failed.");
            return 0;
        }

        // Down projection: [hidden_dim, n_used, seq_len]
        ggml_tensor* down_proj = ggml_mul_mat_id(ctx, down_w, activated, ids_t);
        if (down_proj == nullptr)
        {
            set_last_error("MoE prefill: ggml_mul_mat_id(down) failed.");
            return 0;
        }

        if (down_bias_t != nullptr)
        {
            down_proj = ggml_add_id(ctx, down_proj, down_bias_t, ids_t);
            if (down_proj == nullptr)
            {
                set_last_error("MoE prefill: ggml_add_id(down_bias) failed.");
                return 0;
            }
        }

        // Apply per-(token,expert) routing weights via broadcast multiply.
        ggml_tensor* weighted = ggml_mul(ctx, down_proj, weights_t);
        if (weighted == nullptr)
        {
            set_last_error("MoE prefill: routing weight multiply failed.");
            return 0;
        }

        // Aggregate experts: sum across the n_used dim → [hidden_dim, seq_len].
        ggml_tensor* moe_out = build_expert_aggregation(ctx, weighted, hidden_dim, n_used, seq_len);
        if (moe_out == nullptr)
        {
            set_last_error("MoE prefill: expert aggregation failed.");
            return 0;
        }

        ggml_tensor* output = nullptr;
        if (residual_mode)
        {
            // Gemma 4 MoE post-aggregation chain:
            //   residual += rms_norm(moe_out, eps) * post_ffw_norm_2.weight
            //
            // Mirrors the Gemma4Model.cs MoEForward + RMSNormOp + Ops.Add
            // sequence (`moeOut → postMoeNormed → mlpOut += postMoeNormed`)
            // but as one fused graph dispatch instead of three. The RMSNorm
            // reduces along ne0 (hidden_dim per token) which is exactly the
            // unweighted RMSNorm Gemma 4 expects on the MoE output before
            // the post_ffw_norm_2 weighted scale.
            //
            // ggml_rms_norm operates along the leading dim (ne0); `moe_out`
            // is [hidden_dim, seq_len], so each column (token) is normalised
            // independently.  ggml_mul broadcasts the 1-D post_norm weight
            // across the seq_len dimension, identical semantics to ollama /
            // llama.cpp's RMSNorm-with-weight pattern (ggml_rms_norm + ggml_mul
            // is the canonical idiom because no fused op exists).
            ggml_tensor* moe_normed = ggml_rms_norm(ctx, moe_out, post_norm_eps);
            if (moe_normed == nullptr)
            {
                set_last_error("MoE prefill: post-norm RMSNorm node creation failed.");
                return 0;
            }
            moe_normed = ggml_mul(ctx, moe_normed, post_norm_w_t);
            if (moe_normed == nullptr)
            {
                set_last_error("MoE prefill: post-norm weight multiply failed.");
                return 0;
            }

            // hidden_out_t holds the residual (dense FFN output) value here:
            // it's bound below to `residual_in_out`. The graph reads it,
            // adds the normed MoE result, and writes the sum back over the
            // same backend buffer (in-place residual fold).
            ggml_tensor* added = ggml_add(ctx, hidden_out_t, moe_normed);
            if (added == nullptr)
            {
                set_last_error("MoE prefill: residual add failed.");
                return 0;
            }
            output = ggml_cpy(ctx, added, hidden_out_t);
        }
        else
        {
            output = ggml_cpy(ctx, moe_out, hidden_out_t);
        }
        if (output == nullptr)
        {
            set_last_error("MoE prefill: output copy node creation failed.");
            return 0;
        }
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 1024, false);
        if (graph == nullptr)
        {
            set_last_error("MoE prefill: failed to create graph.");
            return 0;
        }
        ggml_build_forward_expand(graph, output);

        // --- MoE CPU offload: host compute, or accelerator with streamed experts ---
        // An offloaded layer never gets a PERMANENT device copy of its experts.
        // How it is evaluated still depends on the batch: at decode the host
        // wins outright, but for a prefill-sized batch the upload is amortized
        // over every token in it and computing on the accelerator is several
        // times faster end to end. `stream_experts` picks the second; the branch
        // below (taken before any accelerator binding) is the first.
        const int stream_threshold = moe_offload_stream_batch_threshold();
        const bool stream_experts =
            p.run_on_cpu &&
            p.allow_device_stream &&
            stream_threshold > 0 &&
            seq_len >= stream_threshold &&
            g_backend != nullptr;

        if (p.run_on_cpu && !stream_experts)
        {
            MoECpuGraphBinding b;
            b.hidden_t = hidden_t;
            b.hidden_out_t = hidden_out_t;
            b.post_norm_w_t = post_norm_w_t;
            b.gate_w = gate_w;
            b.up_w = up_w;
            b.down_w = down_w;
            b.ids_t = ids_t;
            b.weights_t = weights_t;
            b.gate_bias_t = gate_bias_t;
            b.up_bias_t = up_bias_t;
            b.down_bias_t = down_bias_t;
            b.gate_bias_dim = gate_bias_dim;
            b.hidden_bytes = hidden_bytes;
            b.out_buffer_ptr = residual_mode ? residual_in_out : hidden_out;
            return moe_ffn_execute_on_cpu(ctx, graph, p, b);
        }

        // --- Bind tensors to backend memory ---
        // hidden_in / hidden_out (or residual_in_out): try zero-copy host-pointer
        // mapping first (saves ~hidden_bytes upload + download per call on
        // Metal). The cached host-ptr buffer path reuses the same Metal buffer
        // across calls when the C# Tensor's underlying storage pointer is
        // stable.
        std::vector<BufferHandle> ephem_buffers;
        std::vector<WeightUploadInfo> uploads;
        bool hidden_in_zero_copy = false;
        bool hidden_out_zero_copy = false;
        // The output buffer in residual mode is the residual_in_out pointer;
        // in standalone mode it's hidden_out. Same shape (seq_len*hidden_dim
        // F32) and same kernel write pattern, so the binding code below is
        // identical.
        float* const out_buffer_ptr = residual_mode ? residual_in_out : hidden_out;

        if (dev != nullptr && hidden_bytes >= 4096)
        {
            // cacheable=false: a cached host-ptr buffer cross-pollinates with
            // every other op that touches the same C# Tensor.Storage pointer
            // (because the cache is keyed on the host data pointer, not the
            // ggml_tensor identity), and the resulting Metal buffer reuse
            // degrades scheduler decisions in practice. Pay the small per-call
            // wrap cost; the freed buffers are released by BufferHandle's
            // destructor at the end of the call.
            ggml_backend_buffer_t buf_in = nullptr;
            if (try_get_host_ptr_buffer(g_backend, dev, hidden_in, hidden_bytes, false, buf_in))
            {
                ephem_buffers.emplace_back(buf_in);
                if (ggml_backend_tensor_alloc(buf_in, hidden_t, hidden_in) == GGML_STATUS_SUCCESS)
                {
                    hidden_in_zero_copy = true;
                }
            }

            ggml_backend_buffer_t buf_out = nullptr;
            if (try_get_host_ptr_buffer(g_backend, dev, out_buffer_ptr, hidden_bytes, false, buf_out))
            {
                ephem_buffers.emplace_back(buf_out);
                if (ggml_backend_tensor_alloc(buf_out, hidden_out_t, out_buffer_ptr) == GGML_STATUS_SUCCESS)
                {
                    hidden_out_zero_copy = true;
                }
            }
        }

        // Quantized expert weights: cacheable read-only path (reused across
        // calls), falling back to host-ptr mapping or deferred upload.
        // `stream_experts` skips both shortcuts (see bind_readonly_weight_3d)
        // and records the tensor for the streamed upload below.
        bind_readonly_weight_3d(ctx, g_backend, dev, gate_data,
            static_cast<std::size_t>(gate_total_bytes), gate_w, uploads, ephem_buffers,
            stream_experts);
        if (!fused_gate_up && !single_projection_activation)
        {
            bind_readonly_weight_3d(ctx, g_backend, dev, up_data,
                static_cast<std::size_t>(up_total_bytes), up_w, uploads, ephem_buffers,
                stream_experts);
        }
        bind_readonly_weight_3d(ctx, g_backend, dev, down_data,
            static_cast<std::size_t>(down_total_bytes), down_w, uploads, ephem_buffers,
            stream_experts);

        // Allocate any tensors not bound above (hidden in/out fallback, ids,
        // weights, biases, and all per-graph intermediates) into the persistent,
        // reused gallocr buffer rather than a FRESH device buffer per call. A
        // per-call alloc+free of the multi-hundred-MB MoE scratch fragments VRAM
        // and — on the near-full 16 GB GPU this MoE model runs on — forces the
        // device-resident expert-weight copies to be evicted and re-streamed every
        // forward (the 40×/forward alloc churn was the dominant prefill cost).
        // The reused buffer is grown once to the largest seen graph and kept; it's
        // shared across ops but prefill is serialized under the GPU compute lock.
        // Falls back to per-call allocation when the reuse path is unavailable
        // (TS_GGML_REUSE_COMPUTE_BUF=0 / unsupported backend).
        BufferHandle backend_buffer(nullptr);
        // A streamed offload graph runs BETWEEN two slices of the outer
        // whole-model graph, whose tensors are placed in the shared reuse
        // allocator; re-planning that allocator here would move them out from
        // under the nodes still to come. Use the allocator kept for this path.
        const bool graph_allocated = stream_experts
            ? alloc_graph_moe_stream_gallocr(graph)
            : alloc_graph_reuse_gallocr(graph);
        if (!graph_allocated)
        {
            backend_buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (backend_buffer.value == nullptr)
            {
                set_last_error("MoE prefill: failed to allocate backend buffer for graph tensors.");
                return 0;
            }
        }
        // Drain pending GPU work before the upcoming CPU memcpys so we don't
        // race with in-flight zero-copy GPU writes targeting `hidden_in` /
        // `selected_experts` / `routing_weights` from the previous op.
        host_read_barrier();

        // TS_HOST_MOE_TIMING=1 covers the streamed path too: "how many bytes
        // crossed the bus and how long the graph took" is the whole story of an
        // offloaded prefill, and it is the number to watch when tuning
        // TS_HOST_MOE_DEVICE_MIN_BATCH.
        static const bool s_stream_timing = []() {
            const char* e = std::getenv("TS_HOST_MOE_TIMING");
            return e != nullptr && e[0] == '1';
        }();
        const auto t_upload = std::chrono::steady_clock::now();
        std::size_t streamed_bytes = 0;

        ExpertRuns expert_runs;
        if (stream_experts)
        {
            const std::size_t id_count =
                static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(n_used);
            build_expert_runs(selected_experts, id_count, num_experts, expert_runs);
        }

        for (const WeightUploadInfo& u : uploads)
        {
            if (!u.needs_upload || u.tensor == nullptr || u.host_data == nullptr || u.bytes == 0)
                continue;
            if (stream_experts &&
                (u.tensor == gate_w || u.tensor == up_w || u.tensor == down_w))
            {
                streamed_bytes += stream_expert_weight(
                    g_backend, u.tensor, u.host_data, u.bytes, num_experts, expert_runs);
            }
            else
            {
                ggml_backend_tensor_set(u.tensor, u.host_data, 0, u.bytes);
            }
        }

        if (!hidden_in_zero_copy)
        {
            ggml_backend_tensor_set(hidden_t, hidden_in, 0, hidden_bytes);
        }

        // In residual mode the kernel reads the residual value out of
        // hidden_out_t (so it can add the normed MoE result to it). When the
        // backend buffer is host-mapped we already see the caller's data; if
        // not, we have to push the residual to device first.
        if (residual_mode && !hidden_out_zero_copy)
        {
            ggml_backend_tensor_set(hidden_out_t, residual_in_out, 0, hidden_bytes);
        }

        // post_ffw_norm_2.weight is small (hidden_dim * 4B). Upload directly
        // each call - reusing the cacheable-tensor path for it would require
        // the C# layer to keep the pointer stable across the model's lifetime
        // and to use a separate ggml_tensor identity per layer to avoid
        // cross-layer aliasing in the buffer cache. Easier and just as cheap
        // to push the few KB over the bus on every call.
        if (post_norm_w_t != nullptr)
        {
            ggml_backend_tensor_set(post_norm_w_t, post_norm_w, 0,
                static_cast<std::size_t>(hidden_dim) * sizeof(float));
        }

        // ids and routing weights are always small; copy from C# host arrays.
        ggml_backend_tensor_set(ids_t, selected_experts, 0,
            static_cast<std::size_t>(seq_len) *
            static_cast<std::size_t>(n_used) * sizeof(std::int32_t));
        ggml_backend_tensor_set(weights_t, routing_weights, 0,
            static_cast<std::size_t>(seq_len) *
            static_cast<std::size_t>(n_used) * sizeof(float));

        // Per-expert biases: small (n_experts × ffn_dim or hidden_dim × 4 bytes).
        // Upload directly from the caller's host arrays each call.
        if (gate_bias_t != nullptr)
        {
            ggml_backend_tensor_set(gate_bias_t, gate_bias, 0,
                static_cast<std::size_t>(gate_bias_dim) *
                static_cast<std::size_t>(num_experts) * sizeof(float));
        }
        if (up_bias_t != nullptr)
        {
            ggml_backend_tensor_set(up_bias_t, up_bias, 0,
                static_cast<std::size_t>(n_ff) *
                static_cast<std::size_t>(num_experts) * sizeof(float));
        }
        if (down_bias_t != nullptr)
        {
            ggml_backend_tensor_set(down_bias_t, down_bias, 0,
                static_cast<std::size_t>(hidden_dim) *
                static_cast<std::size_t>(num_experts) * sizeof(float));
        }

        const auto t_compute = std::chrono::steady_clock::now();
        ggml_status status = tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("MoE prefill: ggml_backend_graph_compute failed.");
            return 0;
        }
        if (s_stream_timing && stream_experts)
        {
            const auto t_end = std::chrono::steady_clock::now();
            static thread_local double acc_up = 0.0, acc_gpu = 0.0, acc_mb = 0.0;
            static thread_local int acc_calls = 0;
            acc_up += std::chrono::duration<double, std::milli>(t_compute - t_upload).count();
            acc_gpu += std::chrono::duration<double, std::milli>(t_end - t_compute).count();
            acc_mb += streamed_bytes / 1048576.0;
            if (++acc_calls % 30 == 0)
            {
                std::fprintf(stderr,
                    "[HOSTMOE-STREAM] seq=%d calls=%d experts=%d/%d bytes=%.0f MiB issue=%.1f ms (%.1f GB/s) gpu=%.1f ms pinned=%.1f GiB\n",
                    seq_len, acc_calls, expert_runs.used, num_experts, acc_mb,
                    acc_up, acc_mb / 1024.0 / (acc_up / 1000.0), acc_gpu,
                    host_pinned_bytes() / 1073741824.0);
                std::fflush(stderr);
                acc_up = acc_gpu = acc_mb = 0.0;
            }
        }

        // Result publish. With a zero-copy host-mapped output buffer, the
        // backend has already written into the host-visible memory and we
        // only need to mark pending GPU work for the next host barrier.
        // Otherwise queue an async blit download (Metal) or sync+copy back.
        // out_buffer_ptr is hidden_out (standalone) or residual_in_out
        // (residual mode); the kernel wrote into the same backend tensor
        // (hidden_out_t) in both cases so the finalize path is identical.
        if (hidden_out_zero_copy)
        {
            finalize_compute(true, hidden_out_t, out_buffer_ptr, hidden_bytes);
        }
        else
        {
            finalize_compute_with_download(hidden_out_t, out_buffer_ptr, hidden_bytes);
        }

        clear_last_error();
        return 1;
    }

    // Backwards-compatible thin wrapper: invokes the unified graph builder
    // in standalone mode (writes moe_out to a dedicated buffer, no residual
    // fold). Kept for the existing call sites in C# (Qwen 3.5 / GPT-OSS /
    // the original Gemma 4 MoE non-residual path) that haven't been
    // restructured to pass a residual tensor.
    int moe_ffn_prefill_swiglu_quant_f32_impl(
        float* hidden_in,
        float* hidden_out,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_total_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_total_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_total_bytes,
        const float* gate_bias,
        const float* up_bias,
        const float* down_bias,
        int activation_type,
        float oai_alpha,
        float oai_limit,
        int run_on_cpu)
    {
        MoEFFNGraphParams p;
        p.hidden_in = hidden_in;
        p.hidden_out = hidden_out;
        p.seq_len = seq_len;
        p.hidden_dim = hidden_dim;
        p.n_ff = n_ff;
        p.num_experts = num_experts;
        p.n_used = n_used;
        p.selected_experts = selected_experts;
        p.routing_weights = routing_weights;
        p.gate_data = gate_data;
        p.gate_type = gate_type;
        p.gate_ne0 = gate_ne0;
        p.gate_ne1 = gate_ne1;
        p.gate_total_bytes = gate_total_bytes;
        p.up_data = up_data;
        p.up_type = up_type;
        p.up_ne0 = up_ne0;
        p.up_ne1 = up_ne1;
        p.up_total_bytes = up_total_bytes;
        p.down_data = down_data;
        p.down_type = down_type;
        p.down_ne0 = down_ne0;
        p.down_ne1 = down_ne1;
        p.down_total_bytes = down_total_bytes;
        p.gate_bias = gate_bias;
        p.up_bias = up_bias;
        p.down_bias = down_bias;
        p.activation_type = activation_type;
        p.oai_alpha = oai_alpha;
        p.oai_limit = oai_limit;
        p.run_on_cpu = run_on_cpu != 0;
        // The per-op entry points may stream their experts too, but ONLY
        // outside tensor parallelism: under TP they are driven through
        // RunPerRank, where every rank would stream its own copy of the
        // same unsharded expert layer to its own device at once. A single
        // device is the case that matters here — it is the path GPT-OSS
        // and Nemotron-H take at prefill (no fused whole-model prefill
        // graph, hence no seam), and leaving it host-only cost gpt-oss-20b
        // 4-10x against llama.cpp on every offloaded prefill.
        p.allow_device_stream =
            tsg::g_device_count.load(std::memory_order_acquire) <= 1;
        return moe_ffn_prefill_graph_impl(p);
    }

    // Residual-fused variant: invokes the unified graph builder in residual
    // mode so the caller's residual buffer ends up containing
    // `residual + rms_norm(moe_out, eps) * post_norm_w` after one graph
    // dispatch. Designed for Gemma 4 MoE which routes every layer through
    // dense_FFN + post_norm_1 + (MoE_FFN -> post_norm_2 -> add). The kernel
    // collapses the post_norm_2 + add into the same dispatch as the MoE FFN.
    int gemma4_moe_geglu_residual_f32_impl(
        float* hidden_in,
        float* residual_in_out,
        const float* post_norm_w,
        float post_norm_eps,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_total_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_total_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_total_bytes,
        const float* gate_bias,
        const float* up_bias,
        const float* down_bias,
        int activation_type,
        float oai_alpha,
        float oai_limit,
        int run_on_cpu)
    {
        MoEFFNGraphParams p;
        p.hidden_in = hidden_in;
        p.residual_in_out = residual_in_out;
        p.post_norm_w = post_norm_w;
        p.post_norm_eps = post_norm_eps;
        p.seq_len = seq_len;
        p.hidden_dim = hidden_dim;
        p.n_ff = n_ff;
        p.num_experts = num_experts;
        p.n_used = n_used;
        p.selected_experts = selected_experts;
        p.routing_weights = routing_weights;
        p.gate_data = gate_data;
        p.gate_type = gate_type;
        p.gate_ne0 = gate_ne0;
        p.gate_ne1 = gate_ne1;
        p.gate_total_bytes = gate_total_bytes;
        p.up_data = up_data;
        p.up_type = up_type;
        p.up_ne0 = up_ne0;
        p.up_ne1 = up_ne1;
        p.up_total_bytes = up_total_bytes;
        p.down_data = down_data;
        p.down_type = down_type;
        p.down_ne0 = down_ne0;
        p.down_ne1 = down_ne1;
        p.down_total_bytes = down_total_bytes;
        p.gate_bias = gate_bias;
        p.up_bias = up_bias;
        p.down_bias = down_bias;
        p.activation_type = activation_type;
        p.oai_alpha = oai_alpha;
        p.oai_limit = oai_limit;
        p.run_on_cpu = run_on_cpu != 0;
        // The per-op entry points may stream their experts too, but ONLY
        // outside tensor parallelism: under TP they are driven through
        // RunPerRank, where every rank would stream its own copy of the
        // same unsharded expert layer to its own device at once. A single
        // device is the case that matters here — it is the path GPT-OSS
        // and Nemotron-H take at prefill (no fused whole-model prefill
        // graph, hence no seam), and leaving it host-only cost gpt-oss-20b
        // 4-10x against llama.cpp on every offloaded prefill.
        p.allow_device_stream =
            tsg::g_device_count.load(std::memory_order_acquire) <= 1;
        return moe_ffn_prefill_graph_impl(p);
    }
} // namespace

namespace tsg
{
    // Host-side routed-expert FFN, shared by the standalone MoE op (when the
    // caller passes run_on_cpu) and by the whole-model decode graphs' CPU
    // offload segments. Declared in ggml_ops_internal.h; see there for the
    // buffer contract.
    //
    // Routing is decided by the caller — the whole-model graphs keep the router
    // on the accelerator and hand down the ids/weights they computed — so the
    // host and device paths select the same experts with the same weights and
    // differ only in where the three matmuls run.
    bool moe_ffn_host_experts(
        const float* hidden_in,
        float* hidden_out,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_bytes,
        const float* gate_bias, const float* up_bias, const float* down_bias,
        int activation_type, float oai_alpha, float oai_limit,
        bool allow_device_stream)
    {
        MoEFFNGraphParams p;
        // The graph builder takes a mutable input pointer because the
        // accelerator path may bind it as a zero-copy read-write buffer; the
        // host path only ever reads it.
        p.hidden_in = const_cast<float*>(hidden_in);
        p.hidden_out = hidden_out;
        p.seq_len = seq_len;
        p.hidden_dim = hidden_dim;
        p.n_ff = n_ff;
        p.num_experts = num_experts;
        p.n_used = n_used;
        p.selected_experts = selected_experts;
        p.routing_weights = routing_weights;
        p.gate_data = gate_data;
        p.gate_type = gate_type;
        p.gate_ne0 = gate_ne0;
        p.gate_ne1 = gate_ne1;
        p.gate_total_bytes = gate_bytes;
        p.up_data = up_data;
        p.up_type = up_type;
        p.up_ne0 = up_ne0;
        p.up_ne1 = up_ne1;
        p.up_total_bytes = up_bytes;
        p.down_data = down_data;
        p.down_type = down_type;
        p.down_ne0 = down_ne0;
        p.down_ne1 = down_ne1;
        p.down_total_bytes = down_bytes;
        p.gate_bias = gate_bias;
        p.up_bias = up_bias;
        p.down_bias = down_bias;
        p.activation_type = activation_type;
        p.oai_alpha = oai_alpha;
        p.oai_limit = oai_limit;
        p.run_on_cpu = true;
        p.allow_device_stream = allow_device_stream;
        return moe_ffn_prefill_graph_impl(p) != 0;
    }

    // The worker count the host expert matmul uses by default, so other
    // host-MoE callers (DeepSeek V4 drives its own ggml CPU backend through
    // the scheduler rather than through this seam) inherit the same measured
    // rule instead of re-deriving one.
    int host_moe_default_thread_count()
    {
        return moe_cpu_thread_count();
    }

    int host_moe_explicit_thread_count()
    {
        return g_moe_cpu_threads_override.load(std::memory_order_relaxed);
    }

    void moe_set_host_thread_count(int threads)
    {
        g_moe_cpu_threads_override.store(threads > 0 ? threads : 0, std::memory_order_relaxed);
        // The pool latches its thread count at creation, so rebuild it when one
        // already exists (the flag is parsed before the model loads, but a
        // reload must not keep the previous run's count either).
        std::lock_guard<std::mutex> lock(g_moe_cpu_backend_mutex);
        rebuild_moe_cpu_threadpool_locked();
    }

    void moe_ffn_host_release()
    {
        // Give the locked expert pages back first: a model reload maps a new
        // GGUF, and leaving the previous one pinned would charge its bytes to
        // the pin budget forever.
        host_pin_release_all();
        std::lock_guard<std::mutex> lock(g_moe_cpu_backend_mutex);
        if (g_moe_cpu_backend != nullptr)
        {
            ggml_backend_free(g_moe_cpu_backend);
            g_moe_cpu_backend = nullptr;
        }
        if (g_moe_cpu_threadpool != nullptr)
        {
            ggml_threadpool_free(g_moe_cpu_threadpool);
            g_moe_cpu_threadpool = nullptr;
        }
    }

    // ------------------------------------------------------------------
    // Whole-model decode graph segmentation (see ggml_ops_internal.h)
    // ------------------------------------------------------------------
    bool host_moe_verify_enabled()
    {
        static const bool s_on = []() {
            const char* e = std::getenv("TS_HOST_MOE_VERIFY");
            if (e == nullptr)
                // Kept for the Qwen3.5 debugging sessions this seam was first
                // built against; the new name covers every model.
                e = std::getenv("TS_QWEN35_HOST_MOE_VERIFY");
            return e != nullptr && e[0] == '1';
        }();
        return s_on;
    }

    int host_moe_graph_node_index(ggml_cgraph* graph, ggml_tensor* t)
    {
        if (graph == nullptr || t == nullptr)
            return -1;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            if (ggml_graph_node(graph, i) == t)
                return i;
        return -1;
    }

    bool host_moe_build_segment_ends(
        ggml_cgraph* graph,
        const std::vector<HostMoeSegment>& segments,
        std::vector<int>& seg_end,
        const char* kernel_name)
    {
        seg_end.clear();
        if (segments.empty())
            return true;
        if (graph == nullptr)
        {
            set_last_error(std::string(kernel_name) + ": host-MoE segmentation needs a built graph.");
            return false;
        }

        const int n_nodes = ggml_graph_n_nodes(graph);
        seg_end.reserve(segments.size() + 1);
        for (const HostMoeSegment& hm : segments)
        {
            const int i_in = host_moe_graph_node_index(graph, hm.moe_in);
            const int i_id = host_moe_graph_node_index(graph, hm.sel_ids);
            const int i_w  = host_moe_graph_node_index(graph, hm.weights);
            if (i_in < 0 || i_id < 0 || i_w < 0)
            {
                set_last_error(std::string(kernel_name) + ": host-MoE boundary tensor missing from the graph.");
                seg_end.clear();
                return false;
            }
            int cut = (i_in > i_id ? (i_in > i_w ? i_in : i_w) : (i_id > i_w ? i_id : i_w)) + 1;
            if (hm.verify_gpu != nullptr)
            {
                const int i_v = host_moe_graph_node_index(graph, hm.verify_gpu);
                if (i_v + 1 > cut) cut = i_v + 1;
            }
            if (!seg_end.empty() && cut <= seg_end.back())
            {
                set_last_error(std::string(kernel_name) + ": host-MoE segments are not in graph order.");
                seg_end.clear();
                return false;
            }
            seg_end.push_back(cut);
        }
        seg_end.push_back(n_nodes);
        return true;
    }

    bool host_moe_compute_segment(const HostMoeSegment& hm, std::vector<float>& out, const char* kernel_name,
                                  HostMoeStagedInputs* staged, bool allow_device_stream)
    {
        if (hm.moe_in == nullptr || hm.sel_ids == nullptr ||
            hm.weights == nullptr || hm.moe_out == nullptr)
        {
            set_last_error(std::string(kernel_name) + ": incomplete host-MoE segment binding.");
            return false;
        }

        // Scratch reused across layers, tokens and ranks. The shapes are fixed
        // for a given model, so after the first token these never reallocate.
        static thread_local std::vector<float> s_moe_in;
        static thread_local std::vector<float> s_weights;
        static thread_local std::vector<std::int32_t> s_ids;

        const std::size_t act_count = static_cast<std::size_t>(hm.hidden) * static_cast<std::size_t>(hm.seq_len);
        const std::size_t route_count = static_cast<std::size_t>(hm.n_used) * static_cast<std::size_t>(hm.seq_len);
        s_moe_in.resize(act_count);
        s_ids.resize(route_count);
        s_weights.resize(route_count);
        out.resize(act_count);

        // The caller synchronized the segment that produced these, so the reads
        // see its results.
        ggml_backend_tensor_get(hm.moe_in, s_moe_in.data(), 0, act_count * sizeof(float));
        ggml_backend_tensor_get(hm.sel_ids, s_ids.data(), 0, route_count * sizeof(std::int32_t));
        ggml_backend_tensor_get(hm.weights, s_weights.data(), 0, route_count * sizeof(float));

        if (staged != nullptr)
        {
            staged->moe_in = s_moe_in.data();
            staged->sel_ids = s_ids.data();
            staged->weights = s_weights.data();
        }

        return moe_ffn_host_experts(
            s_moe_in.data(), out.data(),
            hm.seq_len, hm.hidden, hm.n_ff,
            hm.num_experts, hm.n_used,
            s_ids.data(), s_weights.data(),
            hm.gate_data, hm.gate_type, hm.gate_ne0, hm.gate_ne1, hm.gate_bytes,
            hm.up_data,   hm.up_type,   hm.up_ne0,   hm.up_ne1,   hm.up_bytes,
            hm.down_data, hm.down_type, hm.down_ne0, hm.down_ne1, hm.down_bytes,
            hm.gate_bias, hm.up_bias, hm.down_bias,
            hm.activation, hm.oai_alpha, hm.oai_limit,
            allow_device_stream);
    }

    void host_moe_upload_segment(const HostMoeSegment& hm, const float* data)
    {
        const std::size_t act_count = static_cast<std::size_t>(hm.hidden) * static_cast<std::size_t>(hm.seq_len);
        ggml_backend_tensor_set(hm.moe_out, data, 0, act_count * sizeof(float));
    }

    void host_moe_zero_segment(const HostMoeSegment& hm)
    {
        static thread_local std::vector<float> s_zeros;
        const std::size_t act_count = static_cast<std::size_t>(hm.hidden) * static_cast<std::size_t>(hm.seq_len);
        if (s_zeros.size() < act_count)
            s_zeros.assign(act_count, 0.0f);
        ggml_backend_tensor_set(hm.moe_out, s_zeros.data(), 0, act_count * sizeof(float));
    }

    bool host_moe_execute_segments(
        ggml_cgraph* graph,
        const std::vector<HostMoeSegment>& segments,
        const std::vector<int>& seg_end,
        const char* kernel_name)
    {
        if (graph == nullptr || seg_end.empty())
        {
            set_last_error(std::string(kernel_name) + ": empty host-MoE segment plan.");
            return false;
        }

        // One result buffer reused across layers and tokens; the seam helper
        // owns the input staging. Shapes are fixed for a given model, so after
        // the first token neither reallocates.
        static thread_local std::vector<float> s_moe_out;

        // TS_HOST_MOE_DEBUG=1: print the segment plan and each seam's activation
        // norms. An all-zero moe_in means the cut fired before the layer actually
        // produced it — the failure mode this seam is easiest to get wrong in.
        static const bool s_debug = []() {
            const char* e = std::getenv("TS_HOST_MOE_DEBUG");
            return e != nullptr && e[0] == '1';
        }();
        static int s_debug_calls = 0;
        const bool debug_this_call = s_debug && s_debug_calls < 3;
        if (debug_this_call)
        {
            ++s_debug_calls;
            std::fprintf(stderr, "[HOSTMOE-DBG] %s: %d nodes, %d segments, cuts=",
                kernel_name, ggml_graph_n_nodes(graph), (int) segments.size());
            for (int c : seg_end) std::fprintf(stderr, "%d ", c);
            std::fprintf(stderr, "\n");
        }

        int begin = 0;
        for (std::size_t seg = 0; seg < seg_end.size(); ++seg)
        {
            const int end = seg_end[seg];
            if (end > begin)
            {
                ggml_cgraph view = ggml_graph_view(graph, begin, end);
                if (tsg::compute_graph(g_backend, &view) != GGML_STATUS_SUCCESS)
                {
                    set_last_error(std::string(kernel_name) + ": host-MoE segment execution failed.");
                    return false;
                }
            }
            begin = end;

            if (seg >= segments.size())
                continue;

            // ggml_backend_graph_compute above already synchronized, so the
            // helper's reads see this segment's results.
            const HostMoeSegment& hm = segments[seg];

            // Prefill-sized batch: evaluate the layer on the accelerator with
            // its experts streamed in, reading and writing the whole-model
            // graph's own tensors. Nothing to download, nothing to upload.
            // (The verify chain wants the host result to compare against, so it
            // keeps the staged path.)
            HostMoeStagedInputs staged;
            if (!host_moe_compute_segment(hm, s_moe_out, kernel_name, &staged))
                return false;

            const std::size_t act_count = static_cast<std::size_t>(hm.hidden) * static_cast<std::size_t>(hm.seq_len);
            const std::size_t route_count = static_cast<std::size_t>(hm.n_used) * static_cast<std::size_t>(hm.seq_len);

            if (hm.verify_gpu != nullptr)
            {
                static thread_local std::vector<float> s_ref;
                s_ref.resize(act_count);
                ggml_backend_tensor_get(hm.verify_gpu, s_ref.data(), 0, act_count * sizeof(float));
                double num = 0.0, den = 0.0, maxabs = 0.0;
                for (std::size_t i = 0; i < act_count; ++i)
                {
                    const double dlt = (double) s_moe_out[i] - (double) s_ref[i];
                    num += dlt * dlt;
                    den += (double) s_ref[i] * (double) s_ref[i];
                    const double ad = dlt < 0 ? -dlt : dlt;
                    if (ad > maxabs) maxabs = ad;
                }
                static int s_vcalls = 0;
                if (s_vcalls < 24)
                {
                    ++s_vcalls;
                    std::fprintf(stderr,
                        "[HOSTMOE-VERIFY] %s layer=%d relL2=%.5f maxAbsDiff=%.6f host[0..2]=%.5f,%.5f,%.5f gpu[0..2]=%.5f,%.5f,%.5f\n",
                        kernel_name, hm.layer, den > 0 ? std::sqrt(num / den) : 0.0, maxabs,
                        s_moe_out[0], s_moe_out[1], s_moe_out[2],
                        s_ref[0], s_ref[1], s_ref[2]);
                    std::fflush(stderr);
                }
            }

            host_moe_upload_segment(hm, s_moe_out.data());

            if (debug_this_call)
            {
                double nin = 0.0, nout = 0.0;
                for (std::size_t i = 0; i < act_count; ++i)
                {
                    nin += (double) staged.moe_in[i] * staged.moe_in[i];
                    nout += (double) s_moe_out[i] * s_moe_out[i];
                }
                std::fprintf(stderr,
                    "[HOSTMOE-DBG]   seg=%d layer=%d seq=%d |in|=%.4f |out|=%.4f ids[0..3]=%d,%d,%d,%d w[0..1]=%.4f,%.4f\n",
                    (int) seg, hm.layer, hm.seq_len, std::sqrt(nin), std::sqrt(nout),
                    staged.sel_ids[0], route_count > 1 ? staged.sel_ids[1] : -1,
                    route_count > 2 ? staged.sel_ids[2] : -1, route_count > 3 ? staged.sel_ids[3] : -1,
                    staged.weights[0], route_count > 1 ? staged.weights[1] : 0.0f);
                std::fflush(stderr);
            }
        }
        return true;
    }
}

extern "C"
{
    // Worker threads for the host-side MoE matmul (--cpu-moe-threads). 0 =
    // default (available_cpu_parallelism() - 1). Safe to call before any
    // backend exists; it re-applies to a live one.
    TSG_EXPORT void TSGgml_SetHostMoeThreads(int threads)
    {
        try { tsg::moe_set_host_thread_count(threads); }
        catch (...) { /* thread count is advisory; never fail the caller */ }
    }

    // Fused MoE FFN prefill (Mixture-of-Experts SwiGLU FFN).
    //
    // Computes the standard MoE FFN body in a single GGML graph dispatch:
    //   for each token t and used expert u with weight w_{t,u}:
    //     gate = W_gate[expert_id(t,u)] @ hidden[t]
    //     up   = W_up  [expert_id(t,u)] @ hidden[t]   (or split from fused gate_up)
    //     act  = swiglu(gate, up)                     (split or oai variant)
    //     down = W_down[expert_id(t,u)] @ act
    //   moe_out[t] = sum_u w_{t,u} * down_{t,u}
    //
    // hidden_in / hidden_out: contiguous f32 of size seq_len * hidden_dim. The
    //                         caller owns both buffers; the kernel writes only
    //                         `hidden_out` (it does NOT add to a residual).
    // selected_experts:        i32 [seq_len, n_used] in row-major layout
    //                          (selected_experts[t * n_used + u] is the expert
    //                          index used by token t for slot u).
    // routing_weights:         f32 [seq_len, n_used] in the same layout.
    // gate / up / down weights: stacked per-expert quantized blocks with the
    //                          on-disk GGUF layout for `_exps.weight` tensors,
    //                          i.e. [ne0, ne1, num_experts] contiguous. For
    //                          mmap'd models the C# layer passes the base
    //                          pointer of expert 0 for free; otherwise the
    //                          caller stacks them at model load time.
    // up_data == nullptr:      indicates a fused gate_up weight (gate_ne1 must
    //                          be 2 * n_ff), except for activation_type=3
    //                          where gate_data is the single up projection.
    //                          Mirrors llama.cpp's gate_up_exps path where
    //                          supported.
    // activation_type:         0 = silu(gate) * up (regular SwiGLU split)
    //                          1 = swiglu_oai (gpt-oss clamped variant). Uses
    //                              oai_alpha and oai_limit.
    //                          2 = gelu(gate) * up (Gemma 4 MoE GEGLU split).
    //                          3 = relu(up)^2 (Nemotron-H MoE).
    TSG_EXPORT int TSGgml_MoEFFNPrefillSwiGLUQuantF32(
        float* hidden_in,
        float* hidden_out,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_total_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_total_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_total_bytes,
        const float* gate_bias,
        const float* up_bias,
        const float* down_bias,
        int activation_type,
        float oai_alpha,
        float oai_limit,
        int run_on_cpu)
    {
        try
        {
            return moe_ffn_prefill_swiglu_quant_f32_impl(
                hidden_in, hidden_out, seq_len, hidden_dim, n_ff,
                num_experts, n_used, selected_experts, routing_weights,
                gate_data, gate_type, gate_ne0, gate_ne1, gate_total_bytes,
                up_data,   up_type,   up_ne0,   up_ne1,   up_total_bytes,
                down_data, down_type, down_ne0, down_ne1, down_total_bytes,
                gate_bias, up_bias, down_bias,
                activation_type, oai_alpha, oai_limit, run_on_cpu);
        }
        catch (const std::exception& ex)
        {
            set_last_error(ex.what());
            return 0;
        }
        catch (...)
        {
            set_last_error("Unknown error in MoE FFN prefill.");
            return 0;
        }
    }

    // Gemma 4 MoE residual-fused GEGLU kernel.
    //
    // Computes the standard MoE FFN body PLUS the post_ffw_norm_2 RMSNorm
    // (with weighted scale post_norm_w) PLUS the residual add into the
    // dense-FFN-output buffer in a SINGLE GGML graph dispatch:
    //
    //   moe_out      = moe_ffn(hidden_in, gate, up, down, ids, weights)   [batched mul_mat_id]
    //   moe_normed   = rms_norm(moe_out, eps) * post_norm_w
    //   residual_in_out += moe_normed                                     [in-place residual]
    //
    // Saves two device dispatches per MoE layer (the standalone RMSNorm and
    // the standalone Add) versus the previous TryMoEFusedGEGLU + RMSNorm +
    // Add sequence in C#. Mirrors the analogous Qwen 3.5 single-token kernel
    // TSGgml_MoEExpertsSwiGLUResidual which folds the routed-expert + shared-
    // expert + residual into one graph submission.
    //
    // hidden_in (in):           [seq_len, hidden_dim] F32 contiguous - the MoE
    //                           input (pre_ffw_norm_2 output of the residual
    //                           stream in Gemma 4 MoE layers).
    // residual_in_out (in/out): [seq_len, hidden_dim] F32 contiguous - the
    //                           dense FFN output that the kernel adds the
    //                           normed MoE result to. The kernel writes the
    //                           sum back over the same buffer.
    // post_norm_w (in):         [hidden_dim] F32 - the post_ffw_norm_2.weight
    //                           tensor.
    // post_norm_eps:            RMSNorm epsilon (ModelConfig.Eps in C#).
    // selected_experts:         i32 [seq_len, n_used] in row-major layout.
    // routing_weights:          f32 [seq_len, n_used] in row-major layout.
    //                           Per-expert post-down scales (Gemma 4's
    //                           ffn_down_exps.scale) must be folded into
    //                           these by the caller before the call.
    // gate / up / down weights: per-expert stacked quantized blocks, identical
    //                           layout to TSGgml_MoEFFNPrefillSwiGLUQuantF32.
    //                           up_data == nullptr signals a fused gate_up
    //                           weight (gate_ne1 = 2 * n_ff) - the kernel
    //                           splits internally.
    // activation_type:          MoE GLU activation. For Gemma 4 MoE this is
    //                           always 2 (GEGLU split = gelu(gate) * up); the
    //                           parameter is passed through for symmetry with
    //                           the standalone kernel and to leave room for
    //                           future variants without an ABI change.
    TSG_EXPORT int TSGgml_Gemma4MoEGEGLUResidualF32(
        float* hidden_in,
        float* residual_in_out,
        const float* post_norm_w,
        float post_norm_eps,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_total_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_total_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_total_bytes,
        const float* gate_bias,
        const float* up_bias,
        const float* down_bias,
        int activation_type,
        float oai_alpha,
        float oai_limit,
        int run_on_cpu)
    {
        try
        {
            return gemma4_moe_geglu_residual_f32_impl(
                hidden_in, residual_in_out, post_norm_w, post_norm_eps,
                seq_len, hidden_dim, n_ff,
                num_experts, n_used, selected_experts, routing_weights,
                gate_data, gate_type, gate_ne0, gate_ne1, gate_total_bytes,
                up_data,   up_type,   up_ne0,   up_ne1,   up_total_bytes,
                down_data, down_type, down_ne0, down_ne1, down_total_bytes,
                gate_bias, up_bias, down_bias,
                activation_type, oai_alpha, oai_limit, run_on_cpu);
        }
        catch (const std::exception& ex)
        {
            set_last_error(ex.what());
            return 0;
        }
        catch (...)
        {
            set_last_error("Unknown error in Gemma 4 MoE GEGLU residual kernel.");
            return 0;
        }
    }
}
