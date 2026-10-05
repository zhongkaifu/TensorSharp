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
//   phase 1   gate and up rows of every selected expert, in 64-row chunks
//   phase 2   per expert: h = silu(gate) * up, quantized for the down rows
//   phase 3   output rows: sum over experts of weight * dot(down row, h)
//
// Work is handed out by atomic counters, and the calling thread starts on it
// at once, so a worker that wakes late just takes fewer chunks instead of
// holding everyone at a barrier. Outside a call nothing spins.
//
// A bounded batch of independent decode rows shares one team dispatch. Each
// row retains the solo dot products, activation quantization and expert sum
// order. Unsupported widths, biases, a fused gate_up tensor, another activation
// or a type without CPU dot traits return false without writing the output.
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
        constexpr int kMaxRows = 8;
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
            const std::uint8_t* gate[kMaxRows * kMaxUsed] = {};
            const std::uint8_t* up[kMaxRows * kMaxUsed] = {};
            const std::uint8_t* down[kMaxRows * kMaxUsed] = {};
            float weight[kMaxRows * kMaxUsed] = {};
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
            const int e = matrix / 2;
            const bool is_up = (matrix & 1) != 0;
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
            const float* g = j.g + (std::size_t)e * j.n_ff;
            const float* u = j.u + (std::size_t)e * j.n_ff;
            float* h = j.h + (std::size_t)e * j.n_ff;
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
                const std::uint8_t* base = j.down[expert_base + e];
                const void* hq = j.hq + (std::size_t)(expert_base + e) * j.hq_row;
                const float w = j.weight[expert_base + e];
                for (int r = r0; r < r1; ++r)
                {
                    float s = 0.0f;
                    j.down_dot(j.n_ff, &s, 0, base + (std::size_t)r * j.down_row, 0, hq, 0, 1);
                    acc[r - r0] += w * s;
                }
            }
            for (int r = r0; r < r1; ++r) j.out[(std::size_t)row * j.n_embd + r] = acc[r - r0];
        }

        void phase3_project_chunk(DecodeJob& j, int c)
        {
            const int e = j.down_order[c / j.chunks3_per_row];
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

    bool host_moe_decode_experts_rows(const HostMoeSegment& hm, const float* x, const std::int32_t* ids,
                                     const float* weights, float* out)
    {
        if (!decode_kernel_enabled() || hm.seq_len < 1 || hm.seq_len > kMaxRows || hm.activation != 0
            || hm.up_data == nullptr || hm.gate_bias || hm.up_bias || hm.down_bias
            || hm.n_used <= 0 || hm.n_used > kMaxUsed || hm.num_experts <= 0
            || hm.gate_data == nullptr || hm.down_data == nullptr
            || x == nullptr || ids == nullptr || weights == nullptr || out == nullptr)
            return false;
        const int n_rows = hm.seq_len, n_embd = hm.hidden, n_ff = hm.n_ff;
        if (hm.gate_ne0 != n_embd || hm.up_ne0 != n_embd || hm.gate_ne1 != n_ff || hm.up_ne1 != n_ff
            || hm.down_ne0 != n_ff || hm.down_ne1 != n_embd)
            return false;
        const ggml_type_traits_cpu* tg = dot_traits(hm.gate_type);
        const ggml_type_traits_cpu* tu = dot_traits(hm.up_type);
        const ggml_type_traits_cpu* td = dot_traits(hm.down_type);
        if (!tg || !tu || !td
            || !row_fits(hm.gate_type, n_embd) || !row_fits(hm.up_type, n_embd) || !row_fits(hm.down_type, n_ff)
            || !row_fits(tg->vec_dot_type, n_embd) || !row_fits(tu->vec_dot_type, n_embd)
            || !row_fits(td->vec_dot_type, n_ff))
            return false;

        const std::size_t gate_row = ggml_row_size((ggml_type)hm.gate_type, n_embd);
        const std::size_t up_row = ggml_row_size((ggml_type)hm.up_type, n_embd);
        const std::size_t down_row = ggml_row_size((ggml_type)hm.down_type, n_ff);
        const std::size_t gate_stride = gate_row * (std::size_t)n_ff;
        const std::size_t up_stride = up_row * (std::size_t)n_ff;
        const std::size_t down_stride = down_row * (std::size_t)n_embd;
        if ((std::uint64_t)hm.gate_bytes != gate_stride * (std::uint64_t)hm.num_experts
            || (std::uint64_t)hm.up_bytes != up_stride * (std::uint64_t)hm.num_experts
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

        std::lock_guard<std::mutex> guard(g_team_mutex);
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
        for (int k = 0; k < selected; ++k)
        {
            j.gate[k] = (const std::uint8_t*)hm.gate_data + (std::size_t)ids[k] * gate_stride;
            j.up[k] = (const std::uint8_t*)hm.up_data + (std::size_t)ids[k] * up_stride;
            j.down[k] = (const std::uint8_t*)hm.down_data + (std::size_t)ids[k] * down_stride;
            j.weight[k] = weights[k];
        }
        if (n_rows > 1)
        {
            for (int k = 0; k < selected * 2; ++k) j.matrix_order[k] = k;
            std::sort(j.matrix_order, j.matrix_order + selected * 2, [&](int a, int b) {
                const auto* pa = (a & 1) ? j.up[a / 2] : j.gate[a / 2];
                const auto* pb = (b & 1) ? j.up[b / 2] : j.gate[b / 2];
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
        j.chunks1 = selected * 2 * j.chunks_per_matrix;
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
        g_team->run(j);
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
                                 const float* weights, float* out)
    {
        return hm.seq_len == 1 && host_moe_decode_experts_rows(hm, x, ids, weights, out);
    }

    void host_moe_decode_release()
    {
        std::lock_guard<std::mutex> guard(g_team_mutex);
        delete g_team;
        g_team = nullptr;
        g_team_failed = false;
    }
}
