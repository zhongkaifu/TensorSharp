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
// Independent decode-row routed-expert FFN on the host (MoE CPU offload).
//
// An offloaded layer at decode is ten experts' rows - a few tens of MB read
// from the GGUF mapping and a handful of dot products per row. Run as a ggml
// graph on the CPU backend that costs one graph build, five buffer wraps, a
// scratch allocation and a pool kickoff per layer per token, and the kickoff
// is what hurts: ggml's workers sleep between graphs, every graph starts by
// waking them, and the first barrier waits for the slowest one. Measured on an
// M5 Pro (Qwen3.8-Flash-Next, all 48 layers offloaded): 0.6 ms a layer, of
// which ~0.25 ms was the wake-up. Letting the pool spin between graphs halved
// the host side and DOUBLED the accelerator segments in between - the spinning
// cores take the package's shared power and the GPU clocks down.
//
// So this kernel does the same arithmetic with ggml's own CPU dot products
// (ggml_get_type_traits_cpu: the activations quantized to the weight type's
// vec_dot_type, exactly as ggml's mul_mat does), on a team that is woken once
// per layer and parks again as soon as the layer is done:
//
//   phase 1   gate and up rows of every selected expert, in 64-row chunks (up rows only for relu(up)^2)
//   phase 2   per expert: h = silu(gate) * up (or gelu_tanh(gate) * up, or relu(s up)^2), quantized for the down rows
//   phase 3   output rows: sum over experts of weight * dot(down row, h)
//
// Work is handed out by atomic counters, and the calling thread starts on it
// at once, so a worker that wakes late just takes fewer chunks instead of
// holding everyone at a barrier. Outside a call nothing spins.
//
// A bounded batch of independent decode rows shares one team dispatch. Each
// row retains the solo dot products, activation quantization and expert sum
// order. A fused [gate | up] stack (Gemma 4: up_data null, 2 n_ff rows an expert, gate
// first) reads each half as its own matrix; activation 2 is GeGLU (gelu_tanh(gate) * up); activation 3 is
// Nemotron-H's one projection, relu(s up)^2 (the up stack in the gate slot, no up stack; s each route's up scale).
// Unsupported widths, biases, another activation or a type without CPU dot traits return
// false without writing the output.
// ============================================================================

#include "ggml_ops_internal.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace tsg
{
    namespace
    {
        constexpr int kMaxUsed = 64;
        // 16: the direct CUDA engine's batched speculative verify (four sequences' 4-row windows in one step). The
        // ggml callers still pass blocks of 8; a row's arithmetic does not depend on how many rows share the call.
        constexpr int kMaxRows = 16;
        constexpr int kRowsPerChunk = 64;
        constexpr int kSlots = 4;

        // MSVC has neither GCC inline asm nor __builtin_ia32_pause: its intrinsics.
        inline void spin_pause()
        {
#if defined(_MSC_VER) && defined(_M_ARM64)
            __yield();
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
            _mm_pause();
#elif defined(__aarch64__)
            __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        }

        // Everything one call needs. Lives in a team slot so a worker that wakes
        // after the call returned still reads a consistent (finished) job.
        struct DecodeJob
        {
            int n_rows = 0, n_used = 0, n_embd = 0, n_ff = 0;
            int activation = 0;          // 0: silu(gate) * up; 1: GPT-OSS's clamped SwiGLU; 2: gelu_tanh(gate) * up; 3: relu(s up)^2
            int matrices = 2;            // matrices an expert's phase 1 reads: 1 for activation 3 (its up rows, in the gate slot)
            float up_scale[kMaxRows * kMaxUsed] = {};   // activation 3: each route's s
            float swiglu_limit = 0.0f;   // HostMoeSegment::swiglu_limit
            float oai_alpha = 1.702f, oai_limit = 7.0f;   // activation 1
            // Activation 1 (GPT-OSS): each route's expert bias rows (null: none).
            const float* gate_b[kMaxRows * kMaxUsed] = {};
            const float* up_b[kMaxRows * kMaxUsed] = {};
            const float* down_b[kMaxRows * kMaxUsed] = {};
            const std::uint8_t* gate[kMaxRows * kMaxUsed] = {};
            const std::uint8_t* up[kMaxRows * kMaxUsed] = {};
            const std::uint8_t* down[kMaxRows * kMaxUsed] = {};
            float weight[kMaxRows * kMaxUsed] = {};
            // Padding entries (a row with fewer routes than n_used): no work, no
            // contribution. The other entries keep their order and arithmetic.
            bool skip[kMaxRows * kMaxUsed] = {};
            // Multirow-only schedules group repeated mapped matrices by
            // address, without changing each row's logical expert slots.
            int matrix_order[2 * kMaxRows * kMaxUsed] = {};
            int down_order[kMaxRows * kMaxUsed] = {};
            std::size_t gate_row = 0, up_row = 0, down_row = 0;
            ggml_vec_dot_t gate_dot = nullptr, up_dot = nullptr, down_dot = nullptr;
            ggml_from_float_t h_quant = nullptr;
            std::size_t hq_row = 0;
            const void* xq_gate = nullptr;
            const void* xq_up = nullptr;
            std::size_t xq_gate_row = 0, xq_up_row = 0;
            float* g = nullptr;            // [n_rows * n_used * n_ff]
            float* u = nullptr;
            float* h = nullptr;
            std::uint8_t* hq = nullptr;    // [n_rows * n_used * hq_row]
            float* projected = nullptr;   // multirow [n_rows * n_used * n_embd]
            float* out = nullptr;          // [n_rows * n_embd]
            int chunks_per_matrix = 0;     // n_ff / kRowsPerChunk, rounded up
            int chunks1 = 0;               // n_rows * n_used * 2 * chunks_per_matrix
            int chunks3_per_row = 0;
            int chunks3 = 0;
            int chunks4 = 0;

            // TS_HOST_MOE_TIMING=4: when the call started, when the last worker
            // joined, and when each phase completed (steady_clock ns).
            std::int64_t t_start = 0;
            std::atomic<std::int64_t> t_last_join{0};
            std::atomic<int> joined{0};
            std::int64_t t_p1 = 0, t_p2 = 0;
            std::atomic<int> next1{0}, done1{0};
            std::atomic<int> next2{0}, done2{0};
            std::atomic<int> next3{0}, done3{0};
            std::atomic<int> next4{0}, done4{0};
            std::atomic<int> users{0};
        };

        void phase1_chunk(DecodeJob& j, int c)
        {
            const int per = j.chunks_per_matrix;
            const int matrix = j.n_rows == 1 ? c / per : j.matrix_order[c / per];
            const int e = matrix / j.matrices;
            if (j.skip[e]) return;
            const bool is_up = j.matrices == 2 && (matrix & 1) != 0;
            const int r0 = (c % per) * kRowsPerChunk;
            const int r1 = std::min(j.n_ff, r0 + kRowsPerChunk);
            const std::uint8_t* base = is_up ? j.up[e] : j.gate[e];
            const std::size_t row = is_up ? j.up_row : j.gate_row;
            const ggml_vec_dot_t dot = is_up ? j.up_dot : j.gate_dot;
            const void* xq = (is_up ? static_cast<const std::uint8_t*>(j.xq_up) :
                static_cast<const std::uint8_t*>(j.xq_gate)) +
                (std::size_t)(j.n_rows == 1 ? 0 : e / j.n_used) * (is_up ? j.xq_up_row : j.xq_gate_row);
            float* dst = (is_up ? j.u : j.g) + (std::size_t)e * j.n_ff;
            for (int r = r0; r < r1; ++r)
                dot(j.n_embd, dst + r, 0, base + (std::size_t)r * row, 0, xq, 0, 1);
        }

        void phase2_expert(DecodeJob& j, int e)
        {
            if (j.skip[e]) return;
            const float* g = j.g + (std::size_t)e * j.n_ff;
            const float* u = j.u + (std::size_t)e * j.n_ff;
            float* h = j.h + (std::size_t)e * j.n_ff;
            if (j.activation == 1)
            {
                // GPT-OSS (ggml_swiglu_oai): the products plus the expert's biases, the gate clamped from above, the up both ways.
                const float A = j.oai_alpha, L = j.oai_limit;
                const float* gb = j.gate_b[e];
                const float* ub = j.up_b[e];
                for (int i = 0; i < j.n_ff; ++i)
                {
                    const float gi = std::min(g[i] + (gb != nullptr ? gb[i] : 0.0f), L);
                    const float ui = std::min(std::max(u[i] + (ub != nullptr ? ub[i] : 0.0f), -L), L);
                    h[i] = gi / (1.0f + std::exp(-A * gi)) * (ui + 1.0f);
                }
            }
            else if (j.activation == 3)
            {
                // The up product (in the gate slot) times the route's up scale, before the square.
                const float sc = j.up_scale[e];
                for (int i = 0; i < j.n_ff; ++i)
                {
                    const float v = std::max(g[i] * sc, 0.0f);
                    h[i] = v * v;
                }
            }
            else if (j.swiglu_limit > 0.0f)
            {
                const float L = j.swiglu_limit;
                for (int i = 0; i < j.n_ff; ++i)
                {
                    const float gi = std::min(g[i], L), ui = std::min(std::max(u[i], -L), L);
                    h[i] = gi / (1.0f + std::exp(-gi)) * ui;
                }
            }
            else if (j.activation == 2)
                for (int i = 0; i < j.n_ff; ++i)
                {
                    const float gi = g[i];
                    h[i] = 0.5f * gi * (1.0f + std::tanh(0.7978845608028654f * (gi + 0.044715f * gi * gi * gi))) * u[i];
                }
            else
                for (int i = 0; i < j.n_ff; ++i)
                    h[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
            j.h_quant(h, j.hq + (std::size_t)e * j.hq_row, j.n_ff);
        }

        void phase3_chunk(DecodeJob& j, int c)
        {
            const int row = j.n_rows == 1 ? 0 : c / j.chunks3_per_row;
            if (j.n_rows > 1) c %= j.chunks3_per_row;
            const int expert_base = row * j.n_used;
            const int r0 = c * kRowsPerChunk;
            const int r1 = std::min(j.n_embd, r0 + kRowsPerChunk);
            float acc[kRowsPerChunk];
            for (int r = r0; r < r1; ++r) acc[r - r0] = 0.0f;
            // Experts outer, rows inner: each expert's rows are one contiguous run
            // of the mapping. Same summation order as the graph path (expert 0
            // first), so the result does not depend on how chunks were taken.
            for (int e = 0; e < j.n_used; ++e)
            {
                if (j.skip[expert_base + e]) continue;
                const std::uint8_t* base = j.down[expert_base + e];
                const void* hq = j.hq + (std::size_t)(expert_base + e) * j.hq_row;
                const float w = j.weight[expert_base + e];
                const float* db = j.down_b[expert_base + e];
                for (int r = r0; r < r1; ++r)
                {
                    float s = 0.0f;
                    j.down_dot(j.n_ff, &s, 0, base + (std::size_t)r * j.down_row, 0, hq, 0, 1);
                    if (db != nullptr) s += db[r];
                    acc[r - r0] += w * s;
                }
            }
            for (int r = r0; r < r1; ++r) j.out[(std::size_t)row * j.n_embd + r] = acc[r - r0];
        }

        void phase3_project_chunk(DecodeJob& j, int c)
        {
            const int e = j.down_order[c / j.chunks3_per_row];
            if (j.skip[e]) return;
            const int r0 = (c % j.chunks3_per_row) * kRowsPerChunk;
            const int r1 = std::min(j.n_embd, r0 + kRowsPerChunk);
            const auto* base = j.down[e];
            const void* hq = j.hq + (std::size_t)e * j.hq_row;
            float* projected = j.projected + (std::size_t)e * j.n_embd;
            // Walk one mapped expert contiguously before taking another. The
            // solo path instead sums ten matrices within each output chunk;
            // across concurrent rows that churns the mapped working set.
            for (int r = r0; r < r1; ++r)
                j.down_dot(j.n_ff, projected + r, 0,
                    base + (std::size_t)r * j.down_row, 0, hq, 0, 1);
            if (const float* db = j.down_b[e])
                for (int r = r0; r < r1; ++r) projected[r] += db[r];
        }

        void phase4_sum_chunk(DecodeJob& j, int c)
        {
            const int row = c / j.chunks3_per_row;
            const int r0 = (c % j.chunks3_per_row) * kRowsPerChunk;
            const int r1 = std::min(j.n_embd, r0 + kRowsPerChunk);
            const int expert_base = row * j.n_used;
            float acc[kRowsPerChunk];
            for (int r = r0; r < r1; ++r) acc[r - r0] = 0.0f;
            // Preserve the solo arithmetic and original routing-slot order,
            // including duplicate experts and signed routing coefficients.
            for (int e = 0; e < j.n_used; ++e)
            {
                if (j.skip[expert_base + e]) continue;
                const float* projected = j.projected + (std::size_t)(expert_base + e) * j.n_embd;
                const float w = j.weight[expert_base + e];
                for (int r = r0; r < r1; ++r) acc[r - r0] += w * projected[r];
            }
            for (int r = r0; r < r1; ++r) j.out[(std::size_t)row * j.n_embd + r] = acc[r - r0];
        }

        // Take chunks of one phase until none are left, then wait for the ones
        // other participants hold. Returns when the phase is complete.
        template <typename F>
        void run_phase(std::atomic<int>& next, std::atomic<int>& done, int count, F&& body)
        {
            for (;;)
            {
                const int c = next.fetch_add(1, std::memory_order_relaxed);
                if (c >= count) break;
                body(c);
                done.fetch_add(1, std::memory_order_release);
            }
            while (done.load(std::memory_order_acquire) < count)
                spin_pause();
        }

        inline std::int64_t now_ns()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        bool kernel_timing()
        {
            static const bool s_on = []() {
                const char* e = std::getenv("TS_HOST_MOE_TIMING");
                return e != nullptr && e[0] == '4';
            }();
            return s_on;
        }

        void run_job(DecodeJob& j, bool worker)
        {
            if (worker && kernel_timing())
            {
                j.joined.fetch_add(1, std::memory_order_relaxed);
                std::int64_t t = now_ns(), prev = j.t_last_join.load(std::memory_order_relaxed);
                while (t > prev && !j.t_last_join.compare_exchange_weak(prev, t)) {}
            }
            run_phase(j.next1, j.done1, j.chunks1, [&](int c) { phase1_chunk(j, c); });
            if (!worker && kernel_timing()) j.t_p1 = now_ns();
            run_phase(j.next2, j.done2, j.n_rows * j.n_used, [&](int e) { phase2_expert(j, e); });
            if (!worker && kernel_timing()) j.t_p2 = now_ns();
            if (j.n_rows == 1)
                run_phase(j.next3, j.done3, j.chunks3, [&](int c) { phase3_chunk(j, c); });
            else
            {
                run_phase(j.next3, j.done3, j.chunks3, [&](int c) { phase3_project_chunk(j, c); });
                run_phase(j.next4, j.done4, j.chunks4, [&](int c) { phase4_sum_chunk(j, c); });
            }
        }

        class DecodeTeam
        {
        public:
            explicit DecodeTeam(int workers)
            {
                // A thread that fails to start (a process near its thread limit)
                // must not leave the started ones running on a half-built team:
                // stop and join them, then let the caller fall back.
                try
                {
                    for (int i = 0; i < workers; ++i)
                        threads_.emplace_back([this] { worker_loop(); });
                }
                catch (...)
                {
                    shutdown();
                    throw;
                }
            }

            ~DecodeTeam() { shutdown(); }

            int workers() const { return (int)threads_.size(); }

            // Claim the next slot, wait out any straggler still inside its old
            // job, and hand back the slot for the caller to fill. Only the
            // calling thread advances gen_, so reading it here is race-free.
            DecodeJob& acquire()
            {
                DecodeJob& j = slots_[(gen_.load(std::memory_order_relaxed) + 1) % kSlots];
                while (j.users.load(std::memory_order_acquire) != 0)
                    std::this_thread::yield();
                return j;
            }

            // Publish the filled slot, wake the team, and work on it here too.
            void run(DecodeJob& j)
            {
                gen_.fetch_add(1, std::memory_order_seq_cst);
                // Sleepers need the condition variable. The lock orders this
                // notify after any worker that counted itself in and is about to
                // check gen_ under it, so no wake-up is lost.
                if (sleepers_.load(std::memory_order_seq_cst) > 0)
                {
                    {
                        std::lock_guard<std::mutex> lock(mu_);
                    }
                    cv_.notify_all();
                }
                run_job(j, false);
            }

        private:
            void shutdown()
            {
                stop_.store(true, std::memory_order_seq_cst);
                {
                    std::lock_guard<std::mutex> lock(mu_);
                }
                cv_.notify_all();
                for (auto& t : threads_)
                    if (t.joinable()) t.join();
                threads_.clear();
            }

            void worker_loop()
            {
#if defined(__APPLE__)
                // Keep the team on the performance cores; the default QoS can
                // park a woken worker on an efficiency core for a whole layer.
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
                std::uint64_t seen = 0;
                for (;;)
                {
                    if (stop_.load(std::memory_order_acquire)) return;
                    const std::uint64_t g = gen_.load(std::memory_order_acquire);
                    if (g != seen)
                    {
                        DecodeJob& j = slots_[g % kSlots];
                        j.users.fetch_add(1, std::memory_order_acq_rel);
                        // Still the current job? A slot is refilled only kSlots
                        // generations later, and only once its users drain.
                        if (gen_.load(std::memory_order_acquire) != g)
                        {
                            j.users.fetch_sub(1, std::memory_order_acq_rel);
                            continue;
                        }
                        seen = g;
                        run_job(j, true);
                        j.users.fetch_sub(1, std::memory_order_release);
                        continue;
                    }

                    // Sleep until the next layer. Parked threads - spinning, or
                    // even idling in WFE - cost the accelerator its clock: on an
                    // M5 Pro either one took the segments between layers from
                    // 38.6 to 59-62 ms a token, far more than the ~80 us wake-up
                    // saves. So the team sleeps, and late wakers just take fewer
                    // chunks (see run_phase).
                    std::unique_lock<std::mutex> lock(mu_);
                    sleepers_.fetch_add(1, std::memory_order_seq_cst);
                    cv_.wait(lock, [&] {
                        return stop_.load(std::memory_order_seq_cst)
                            || gen_.load(std::memory_order_seq_cst) != seen;
                    });
                    sleepers_.fetch_sub(1, std::memory_order_seq_cst);
                }
            }

            std::vector<std::thread> threads_;
            std::mutex mu_;
            std::condition_variable cv_;
            alignas(128) std::atomic<std::uint64_t> gen_{0};
            alignas(128) std::atomic<int> sleepers_{0};
            std::atomic<bool> stop_{false};
            DecodeJob slots_[kSlots];
        };

        std::mutex g_team_mutex;          // one offloaded layer at a time
        // Telemetry totals (tensorsharp.moe.host.compute.time and the host route count).
        std::atomic<long long> g_rows_ns{0}, g_rows_calls{0}, g_rows_slots{0};
        DecodeTeam* g_team = nullptr;
        bool g_team_failed = false;       // the team could not start; use the graph path

        bool decode_kernel_enabled()
        {
            static const bool s_on = []() {
                const char* e = std::getenv("TS_HOST_MOE_DECODE");
                return !(e != nullptr && e[0] == '0');
            }();
            return s_on;
        }

        const ggml_type_traits_cpu* dot_traits(int type)
        {
            if (type < 0 || type >= GGML_TYPE_COUNT) return nullptr;
            const ggml_type_traits_cpu* t = ggml_get_type_traits_cpu((ggml_type)type);
            if (t == nullptr || t->vec_dot == nullptr) return nullptr;
            const ggml_type_traits_cpu* q = ggml_get_type_traits_cpu(t->vec_dot_type);
            if (q == nullptr || (t->vec_dot_type != GGML_TYPE_F32 && q->from_float == nullptr)) return nullptr;
            return t;
        }

        bool row_fits(int type, std::int64_t n)
        {
            return n > 0 && n % ggml_blck_size((ggml_type)type) == 0;
        }
    }

    // Under g_team_mutex (the resolver's lock): read `a`'s slices ahead.
    static void prefetch_slices_locked(const HostMoePrefetch& a)
    {
        constexpr int kMaxPrefetch = 3 * kMaxRows * kMaxUsed;
        const HostMoeSegment& hm = *a.hm;
        if (a.ids == nullptr || a.count <= 0 || hm.num_experts <= 0 || hm.hidden <= 0 || hm.n_ff <= 0) return;
        // A fused [gate | up] stack (up_data null) holds both halves in the gate stack: one slice an expert. The one
        // projection (activation 3) holds the up rows in the gate slot.
        const bool one = hm.activation == 3;
        const bool fused = !one && hm.up_data == nullptr;
        const std::size_t gate_stride = ggml_row_size((ggml_type)hm.gate_type, hm.hidden) * (std::size_t)hm.n_ff * (fused ? 2 : 1);
        const std::size_t up_stride = fused ? 0 : ggml_row_size((ggml_type)hm.up_type, hm.hidden) * (std::size_t)hm.n_ff;
        const std::size_t down_stride = ggml_row_size((ggml_type)hm.down_type, hm.n_ff) * (std::size_t)hm.hidden;
        const void* sources[kMaxPrefetch];
        std::size_t lengths[kMaxPrefetch];
        int n = 0;
        for (int k = 0; k < a.count && n + 3 <= kMaxPrefetch; ++k)
        {
            if (a.ids[k] < 0 || a.ids[k] >= hm.num_experts) continue;
            sources[n] = (const std::uint8_t*)hm.gate_data + (std::size_t)a.ids[k] * gate_stride; lengths[n++] = gate_stride;
            if (!fused && !one) { sources[n] = (const std::uint8_t*)hm.up_data + (std::size_t)a.ids[k] * up_stride; lengths[n++] = up_stride; }
            sources[n] = (const std::uint8_t*)hm.down_data + (std::size_t)a.ids[k] * down_stride; lengths[n++] = down_stride;
        }
        host_expert_io_prefetch(sources, lengths, n);
    }

    bool host_moe_decode_experts_rows(const HostMoeSegment& hm, const float* x, const std::int32_t* ids,
                                     const float* weights, float* out, const std::uint8_t* skip,
                                     const HostMoePrefetch* ahead, const float* up_scales)
    {
        // The one projection (activation 3, Nemotron-H): the up stack in the gate slot, no up stack.
        const bool one = hm.activation == 3;
        // A fused [gate | up] stack (Gemma 4): no up stack, the gate stack's experts 2 n_ff rows each, gate first.
        const bool fused = !one && hm.up_data == nullptr;
        // Activation 1 (GPT-OSS's clamped SwiGLU) over separate gate and up stacks, the experts' biases with it only.
        const bool oai = hm.activation == 1;
        if (!decode_kernel_enabled() || hm.seq_len < 1 || hm.seq_len > kMaxRows
            || (hm.activation != 0 && hm.activation != 1 && hm.activation != 2 && hm.activation != 3) || (one && hm.up_data != nullptr)
            || (up_scales != nullptr && !one) || (oai && hm.up_data == nullptr)
            || (hm.activation != 0 && hm.swiglu_limit > 0.0f) || (!oai && (hm.gate_bias || hm.up_bias || hm.down_bias))
            || hm.n_used <= 0 || hm.n_used > kMaxUsed || hm.num_experts <= 0
            || hm.gate_data == nullptr || hm.down_data == nullptr
            || x == nullptr || ids == nullptr || weights == nullptr || out == nullptr)
            return false;
        const int n_rows = hm.seq_len, n_embd = hm.hidden, n_ff = hm.n_ff;
        if (hm.gate_ne0 != n_embd || hm.gate_ne1 != (fused ? 2 * (std::int64_t)n_ff : n_ff)
            || (!fused && !one && (hm.up_ne0 != n_embd || hm.up_ne1 != n_ff))
            || hm.down_ne0 != n_ff || hm.down_ne1 != n_embd)
            return false;
        const int up_type = fused || one ? hm.gate_type : hm.up_type;
        const ggml_type_traits_cpu* tg = dot_traits(hm.gate_type);
        const ggml_type_traits_cpu* tu = dot_traits(up_type);
        const ggml_type_traits_cpu* td = dot_traits(hm.down_type);
        if (!tg || !tu || !td
            || !row_fits(hm.gate_type, n_embd) || !row_fits(up_type, n_embd) || !row_fits(hm.down_type, n_ff)
            || !row_fits(tg->vec_dot_type, n_embd) || !row_fits(tu->vec_dot_type, n_embd)
            || !row_fits(td->vec_dot_type, n_ff))
            return false;

        const std::size_t gate_row = ggml_row_size((ggml_type)hm.gate_type, n_embd);
        const std::size_t up_row = ggml_row_size((ggml_type)up_type, n_embd);
        const std::size_t down_row = ggml_row_size((ggml_type)hm.down_type, n_ff);
        // One expert's gate (and, fused, its up rows right after) and its up rows: each half its own slice.
        const std::size_t gate_stride = gate_row * (std::size_t)n_ff * (fused ? 2 : 1);
        const std::size_t half = gate_row * (std::size_t)n_ff;
        const std::size_t up_stride = up_row * (std::size_t)n_ff;
        const std::size_t down_stride = down_row * (std::size_t)n_embd;
        if ((std::uint64_t)hm.gate_bytes != gate_stride * (std::uint64_t)hm.num_experts
            || (!fused && !one && (std::uint64_t)hm.up_bytes != up_stride * (std::uint64_t)hm.num_experts)
            || (std::uint64_t)hm.down_bytes != down_stride * (std::uint64_t)hm.num_experts)
            return false;
        const int selected = n_rows * hm.n_used;
        // Keep task counters bounded before any scratch resize or dispatch.
        if (n_ff > std::numeric_limits<int>::max() - kRowsPerChunk ||
            n_embd > std::numeric_limits<int>::max() - kRowsPerChunk ||
            (n_ff + kRowsPerChunk - 1) / kRowsPerChunk >
                std::numeric_limits<int>::max() / (selected * 2) ||
            (n_embd + kRowsPerChunk - 1) / kRowsPerChunk >
                std::numeric_limits<int>::max() / selected)
            return false;
        for (int k = 0; k < selected; ++k)
            if (ids[k] < 0 || ids[k] >= hm.num_experts) return false;

        // Every routed slice, all its parts (two for the one projection), resolved in one call so the
        // explicitly read cache fetches all misses together.
        const int parts = one ? 2 : 3;
        const void* sources[3 * kMaxRows * kMaxUsed];
        std::size_t lengths[3 * kMaxRows * kMaxUsed];
        const std::uint8_t* resolved[3 * kMaxRows * kMaxUsed];
        for (int k = 0; k < selected; ++k)
        {
            const std::uint8_t* gate_at = (const std::uint8_t*)hm.gate_data + (std::size_t)ids[k] * gate_stride;
            sources[parts * k] = gate_at;
            lengths[parts * k] = half;
            if (!one)
            {
                sources[parts * k + 1] = fused ? gate_at + half : (const std::uint8_t*)hm.up_data + (std::size_t)ids[k] * up_stride;
                lengths[parts * k + 1] = up_stride;
            }
            sources[parts * k + parts - 1] = (const std::uint8_t*)hm.down_data + (std::size_t)ids[k] * down_stride;
            lengths[parts * k + parts - 1] = down_stride;
        }

        std::lock_guard<std::mutex> guard(g_team_mutex);
        // Under the team lock: the slots must stay put until this call's compute
        // is done, and every caller of the resolver holds this lock.
        host_expert_io_resolve(sources, lengths, parts * selected, resolved);
        if (ahead != nullptr && ahead->hm != nullptr) prefetch_slices_locked(*ahead);
        if (g_team == nullptr)
        {
            if (g_team_failed)
                return false;
            ggml_cpu_init();
            const int threads = std::max(1, host_moe_default_thread_count());
            try
            {
                g_team = new DecodeTeam(threads - 1);
            }
            catch (const std::exception& e)
            {
                g_team_failed = true;
                std::fprintf(stderr, "[host-moe] the one-token decode team could not start %d thread(s) (%s); "
                    "offloaded layers decode on the ggml graph path\n", threads - 1, e.what());
                return false;
            }
        }

        // Scratch kept across calls; the shapes are fixed for a model.
        static std::vector<std::uint8_t> s_xq_gate, s_xq_up, s_hq;
        static std::vector<float> s_g, s_u, s_h, s_projected;
        const ggml_type q_gate = tg->vec_dot_type, q_up = tu->vec_dot_type, q_down = td->vec_dot_type;
        const std::size_t xq_gate_row = ggml_row_size(q_gate, n_embd);
        const std::size_t xq_up_row = ggml_row_size(q_up, n_embd);
        s_xq_gate.resize(xq_gate_row * n_rows);
        const auto quant_gate = ggml_get_type_traits_cpu(q_gate)->from_float;
        for (int r = 0; r < n_rows; ++r)
            quant_gate(x + (std::size_t)r * n_embd, s_xq_gate.data() + (std::size_t)r * xq_gate_row, n_embd);
        const void* xq_up = s_xq_gate.data();
        if (q_up != q_gate)
        {
            s_xq_up.resize(xq_up_row * n_rows);
            const auto quant_up = ggml_get_type_traits_cpu(q_up)->from_float;
            for (int r = 0; r < n_rows; ++r)
                quant_up(x + (std::size_t)r * n_embd, s_xq_up.data() + (std::size_t)r * xq_up_row, n_embd);
            xq_up = s_xq_up.data();
        }
        const std::size_t hq_row = ggml_row_size(q_down, n_ff);
        s_g.resize((std::size_t)selected * n_ff);
        s_u.resize((std::size_t)selected * n_ff);
        s_h.resize((std::size_t)selected * n_ff);
        s_hq.resize((std::size_t)selected * hq_row);
        if (n_rows > 1) s_projected.resize((std::size_t)selected * n_embd);

        DecodeJob& j = g_team->acquire();
        j.n_rows = n_rows; j.n_used = hm.n_used; j.n_embd = n_embd; j.n_ff = n_ff;
        j.swiglu_limit = hm.swiglu_limit;
        j.activation = hm.activation;
        j.oai_alpha = hm.oai_alpha;
        j.oai_limit = hm.oai_limit;
        j.matrices = one ? 1 : 2;
        for (int k = 0; k < selected; ++k)
        {
            j.gate_b[k] = oai && hm.gate_bias != nullptr ? hm.gate_bias + (std::size_t)ids[k] * n_ff : nullptr;
            j.up_b[k] = oai && hm.up_bias != nullptr ? hm.up_bias + (std::size_t)ids[k] * n_ff : nullptr;
            j.down_b[k] = oai && hm.down_bias != nullptr ? hm.down_bias + (std::size_t)ids[k] * n_embd : nullptr;
            j.gate[k] = resolved[parts * k];
            j.up[k] = one ? nullptr : resolved[parts * k + 1];
            j.down[k] = resolved[parts * k + parts - 1];
            j.weight[k] = weights[k];
            j.up_scale[k] = up_scales != nullptr ? up_scales[k] : 1.0f;
            j.skip[k] = skip != nullptr && skip[k] != 0;
        }
        if (n_rows > 1)
        {
            const int m = j.matrices;
            for (int k = 0; k < selected * m; ++k) j.matrix_order[k] = k;
            std::sort(j.matrix_order, j.matrix_order + selected * m, [&](int a, int b) {
                const auto* pa = (m == 2 && (a & 1)) ? j.up[a / m] : j.gate[a / m];
                const auto* pb = (m == 2 && (b & 1)) ? j.up[b / m] : j.gate[b / m];
                const auto aa = reinterpret_cast<std::uintptr_t>(pa), bb = reinterpret_cast<std::uintptr_t>(pb);
                return aa < bb || (aa == bb && a < b);
            });
            for (int k = 0; k < selected; ++k) j.down_order[k] = k;
            std::sort(j.down_order, j.down_order + selected, [&](int a, int b) {
                const auto aa = reinterpret_cast<std::uintptr_t>(j.down[a]);
                const auto bb = reinterpret_cast<std::uintptr_t>(j.down[b]);
                return aa < bb || (aa == bb && a < b);
            });
        }
        j.gate_row = gate_row; j.up_row = up_row; j.down_row = down_row;
        j.gate_dot = tg->vec_dot; j.up_dot = tu->vec_dot; j.down_dot = td->vec_dot;
        j.h_quant = ggml_get_type_traits_cpu(q_down)->from_float;
        j.hq_row = hq_row;
        j.xq_gate = s_xq_gate.data(); j.xq_up = xq_up;
        j.xq_gate_row = xq_gate_row; j.xq_up_row = xq_up_row;
        j.g = s_g.data(); j.u = s_u.data(); j.h = s_h.data(); j.hq = s_hq.data();
        j.out = out; j.projected = n_rows > 1 ? s_projected.data() : nullptr;
        j.chunks_per_matrix = (n_ff + kRowsPerChunk - 1) / kRowsPerChunk;
        j.chunks1 = selected * j.matrices * j.chunks_per_matrix;
        j.chunks3_per_row = (n_embd + kRowsPerChunk - 1) / kRowsPerChunk;
        j.chunks3 = n_rows == 1 ? j.chunks3_per_row : selected * j.chunks3_per_row;
        j.chunks4 = n_rows * j.chunks3_per_row;
        j.next1.store(0, std::memory_order_relaxed); j.done1.store(0, std::memory_order_relaxed);
        j.next2.store(0, std::memory_order_relaxed); j.done2.store(0, std::memory_order_relaxed);
        j.next3.store(0, std::memory_order_relaxed); j.done3.store(0, std::memory_order_relaxed);
        j.next4.store(0, std::memory_order_relaxed); j.done4.store(0, std::memory_order_relaxed);
        const bool timing = kernel_timing();
        if (timing)
        {
            j.t_start = now_ns();
            j.t_last_join.store(j.t_start, std::memory_order_relaxed);
            j.joined.store(0, std::memory_order_relaxed);
        }
        const std::int64_t t_run = now_ns();
        g_team->run(j);
        {
            int computed = 0;
            for (int k = 0; k < selected; ++k) computed += j.skip[k] ? 0 : 1;
            g_rows_ns.fetch_add(now_ns() - t_run, std::memory_order_relaxed);
            g_rows_calls.fetch_add(1, std::memory_order_relaxed);
            g_rows_slots.fetch_add(computed, std::memory_order_relaxed);
        }
        if (timing)
        {
            static double acc_total = 0, acc_join = 0, acc_p1 = 0, acc_p2 = 0, acc_joined = 0;
            static int calls = 0;
            const std::int64_t t_end = now_ns();
            acc_total += (t_end - j.t_start) / 1e3;
            acc_join += (j.t_last_join.load() - j.t_start) / 1e3;
            acc_p1 += (j.t_p1 - j.t_start) / 1e3;
            acc_p2 += (j.t_p2 - j.t_p1) / 1e3;
            acc_joined += j.joined.load();
            if (++calls == 480)
            {
                std::fprintf(stderr, "[HOSTMOE-KERNEL] %d workers: %.0f us a call = phase1 %.0f + phase2 %.0f + down/sum %.0f; "
                    "last worker joined after %.0f us (%.1f of %d joined in time)\n",
                    g_team->workers(), acc_total / calls, acc_p1 / calls, acc_p2 / calls,
                    (acc_total - acc_p1 - acc_p2) / calls, acc_join / calls, acc_joined / calls, g_team->workers());
                acc_total = acc_join = acc_p1 = acc_p2 = acc_joined = 0;
                calls = 0;
            }
        }
        return true;
    }

    bool host_moe_decode_experts(const HostMoeSegment& hm, const float* x, const std::int32_t* ids,
                                 const float* weights, float* out, const std::uint8_t* skip, const float* up_scales)
    {
        return hm.seq_len == 1 && host_moe_decode_experts_rows(hm, x, ids, weights, out, skip, nullptr, up_scales);
    }

    void host_moe_decode_counters(long long* ns, long long* calls, long long* slots)
    {
        if (ns) *ns = g_rows_ns.load(std::memory_order_relaxed);
        if (calls) *calls = g_rows_calls.load(std::memory_order_relaxed);
        if (slots) *slots = g_rows_slots.load(std::memory_order_relaxed);
    }

    void host_moe_decode_release()
    {
        std::lock_guard<std::mutex> guard(g_team_mutex);
        delete g_team;
        g_team = nullptr;
        g_team_failed = false;
    }
}

// Routed experts of up to 16 independent rows on the host, from raw expert stacks: the direct CUDA
// engine's host experts (expert residency) and the tensor executors' expert host, where the stacks are
// pinned host copies of the experts without a VRAM slot or the model's mapping, in expert order (ids index
// the stacks; skip[k] != 0 leaves slot k to the device). up == nullptr: gate is a fused [gate | up] stack
// (2 n_ff rows an expert, gate first). activation 0: silu(gate) * up; 2: gelu_tanh(gate) * up. The same
// kernel, dot products and summation order as an offloaded ggml layer. Returns 1 when out[rows, hidden]
// was written, 0 for a shape or type the kernel does not cover.
TSG_EXPORT int TSGgml_HostExpertRowsAct(const void* gate, const void* up, const void* down,
                                        int gate_type, int up_type, int down_type,
                                        int num_experts, int hidden, int n_ff, int n_used, int rows,
                                        const float* x, const std::int32_t* ids, const float* weights,
                                        const std::uint8_t* skip, float* out, int activation)
{
    const bool fused = up == nullptr;
    if (gate == nullptr || down == nullptr || num_experts <= 0 || hidden <= 0 || n_ff <= 0
        || n_used <= 0 || rows <= 0 || gate_type < 0 || gate_type >= GGML_TYPE_COUNT
        || (!fused && (up_type < 0 || up_type >= GGML_TYPE_COUNT)) || down_type < 0 || down_type >= GGML_TYPE_COUNT
        || (activation != 0 && activation != 2))
        return 0;
    tsg::HostMoeSegment hm;
    hm.gate_data = const_cast<void*>(gate);
    hm.up_data = const_cast<void*>(up);
    hm.down_data = const_cast<void*>(down);
    hm.gate_type = gate_type; hm.up_type = fused ? 0 : up_type; hm.down_type = down_type;
    hm.gate_ne0 = hidden; hm.gate_ne1 = fused ? 2 * (std::int64_t)n_ff : n_ff;
    hm.up_ne0 = fused ? 0 : hidden; hm.up_ne1 = fused ? 0 : n_ff;
    hm.down_ne0 = n_ff; hm.down_ne1 = hidden;
    hm.gate_bytes = (std::int64_t)(ggml_row_size((ggml_type)gate_type, hidden) * (std::size_t)hm.gate_ne1) * num_experts;
    hm.up_bytes = fused ? 0 : (std::int64_t)(ggml_row_size((ggml_type)up_type, hidden) * (std::size_t)n_ff) * num_experts;
    hm.down_bytes = (std::int64_t)(ggml_row_size((ggml_type)down_type, n_ff) * (std::size_t)hidden) * num_experts;
    hm.activation = activation;
    hm.num_experts = num_experts;
    hm.n_used = n_used;
    hm.seq_len = rows;
    hm.independent_decode_rows = true;
    hm.hidden = hidden;
    hm.n_ff = n_ff;
    return tsg::host_moe_decode_experts_rows(hm, x, ids, weights, out, skip, nullptr) ? 1 : 0;
}

// GPT-OSS's biased experts on the host, on the one-token decode kernel (independent rows, no ggml backend needed: the
// tensor executors' expert host): gate and up products plus each expert's biases, ggml_swiglu_oai at alpha and limit, the
// down product plus its bias (left out where down_bias is null), weighted. Biases F32 [n_ff, num_experts] (gate, up) and
// [hidden, num_experts] (down). A skipped slot (skip[k] != 0, or a negative id) contributes nothing. Returns 1 when
// out[rows, hidden] was written, 0 (nothing written) for a layout the decode kernel does not take.
TSG_EXPORT int TSGgml_HostExpertRowsOai(const void* gate, const void* up, const void* down,
                                        int gate_type, int up_type, int down_type,
                                        int num_experts, int hidden, int n_ff, int n_used, int rows,
                                        const float* x, const std::int32_t* ids, const float* weights,
                                        const std::uint8_t* skip, float* out,
                                        const float* gate_bias, const float* up_bias, const float* down_bias,
                                        float alpha, float limit)
{
    if (gate == nullptr || up == nullptr || down == nullptr || num_experts <= 0 || hidden <= 0 || n_ff <= 0
        || n_used <= 0 || rows <= 0 || gate_type < 0 || gate_type >= GGML_TYPE_COUNT || up_type < 0 || up_type >= GGML_TYPE_COUNT
        || down_type < 0 || down_type >= GGML_TYPE_COUNT)
        return 0;
    const std::size_t routes = static_cast<std::size_t>(rows) * n_used;
    std::vector<std::int32_t> slot_ids(ids, ids + routes);
    std::vector<float> slot_weights(weights, weights + routes);
    for (std::size_t i = 0; i < routes; i++)
        if ((skip != nullptr && skip[i] != 0) || slot_ids[i] < 0 || slot_ids[i] >= num_experts)
        {
            slot_ids[i] = 0;
            slot_weights[i] = 0.0f;
        }
    const std::int64_t gate_bytes = (std::int64_t)(ggml_row_size((ggml_type)gate_type, hidden) * (std::size_t)n_ff) * num_experts;
    const std::int64_t up_bytes = (std::int64_t)(ggml_row_size((ggml_type)up_type, hidden) * (std::size_t)n_ff) * num_experts;
    const std::int64_t down_bytes = (std::int64_t)(ggml_row_size((ggml_type)down_type, n_ff) * (std::size_t)hidden) * num_experts;
    tsg::HostMoeSegment hm;
    hm.gate_data = const_cast<void*>(gate); hm.up_data = const_cast<void*>(up); hm.down_data = const_cast<void*>(down);
    hm.gate_type = gate_type; hm.up_type = up_type; hm.down_type = down_type;
    hm.gate_ne0 = hidden; hm.gate_ne1 = n_ff; hm.up_ne0 = hidden; hm.up_ne1 = n_ff; hm.down_ne0 = n_ff; hm.down_ne1 = hidden;
    hm.gate_bytes = gate_bytes; hm.up_bytes = up_bytes; hm.down_bytes = down_bytes;
    hm.gate_bias = gate_bias; hm.up_bias = up_bias; hm.down_bias = down_bias;
    hm.activation = 1; hm.oai_alpha = alpha; hm.oai_limit = limit;
    hm.num_experts = num_experts; hm.n_used = n_used; hm.seq_len = rows; hm.independent_decode_rows = true;
    hm.hidden = hidden; hm.n_ff = n_ff;
    std::vector<std::uint8_t> skips(routes, 0);
    for (std::size_t i = 0; i < routes; i++) skips[i] = slot_weights[i] == 0.0f && slot_ids[i] == 0 ? 1 : 0;
    return tsg::host_moe_decode_experts_rows(hm, x, slot_ids.data(), slot_weights.data(), out, skips.data(), nullptr) ? 1 : 0;
}

// The same for Nemotron-H's one-projection experts: relu(s up)^2, then down; `up` is the up stack (n_ff rows an expert),
// `up_scales` each route's s ([rows, n_used] as weights; null for 1).
TSG_EXPORT int TSGgml_HostExpertRowsReluSquared(const void* up, const void* down, int up_type, int down_type,
                                                int num_experts, int hidden, int n_ff, int n_used, int rows,
                                                const float* x, const std::int32_t* ids, const float* weights,
                                                const float* up_scales, const std::uint8_t* skip, float* out)
{
    if (up == nullptr || down == nullptr || num_experts <= 0 || hidden <= 0 || n_ff <= 0 || n_used <= 0 || rows <= 0
        || up_type < 0 || up_type >= GGML_TYPE_COUNT || down_type < 0 || down_type >= GGML_TYPE_COUNT)
        return 0;
    tsg::HostMoeSegment hm;
    hm.gate_data = const_cast<void*>(up);   // the one projection sits in the gate slot
    hm.up_data = nullptr;
    hm.down_data = const_cast<void*>(down);
    hm.gate_type = up_type; hm.up_type = 0; hm.down_type = down_type;
    hm.gate_ne0 = hidden; hm.gate_ne1 = n_ff;
    hm.up_ne0 = 0; hm.up_ne1 = 0;
    hm.down_ne0 = n_ff; hm.down_ne1 = hidden;
    hm.gate_bytes = (std::int64_t)(ggml_row_size((ggml_type)up_type, hidden) * (std::size_t)n_ff) * num_experts;
    hm.up_bytes = 0;
    hm.down_bytes = (std::int64_t)(ggml_row_size((ggml_type)down_type, n_ff) * (std::size_t)hidden) * num_experts;
    hm.activation = 3;
    hm.num_experts = num_experts;
    hm.n_used = n_used;
    hm.seq_len = rows;
    hm.independent_decode_rows = true;
    hm.hidden = hidden;
    hm.n_ff = n_ff;
    return tsg::host_moe_decode_experts_rows(hm, x, ids, weights, out, skip, nullptr, up_scales) ? 1 : 0;
}

TSG_EXPORT int TSGgml_HostExpertRows(const void* gate, const void* up, const void* down,
                                     int gate_type, int up_type, int down_type,
                                     int num_experts, int hidden, int n_ff, int n_used, int rows,
                                     const float* x, const std::int32_t* ids, const float* weights,
                                     const std::uint8_t* skip, float* out)
{
    if (up == nullptr) return 0;
    return TSGgml_HostExpertRowsAct(gate, up, down, gate_type, up_type, down_type, num_experts, hidden, n_ff, n_used, rows,
                                    x, ids, weights, skip, out, 0);
}
