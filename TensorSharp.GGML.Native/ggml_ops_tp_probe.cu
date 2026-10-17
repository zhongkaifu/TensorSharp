// Functional pre-flight probe of the CUDA device collective used by tensor
// parallelism.
//
// Motivation: on some multi-GPU hosts — virtualized cloud instances in
// particular — CUDA peer-to-peer is *advertised* as available (the driver
// reports P2P capability, `nvidia-smi topo -p2p` prints OK, NCCL's topology
// detection concludes `intraNodeP2pSupport 1`) but peer traffic never actually
// completes. NCCL's communicator init succeeds in well under a second, and the
// first collective then enqueues kernels that spin forever waiting for peer
// data that never arrives. Both GPUs sit at "100% utilization" in the spin
// kernels and the host thread spins in stream polling, so a model load that
// should take seconds appears hung for tens of minutes (observed with 2x A40
// behind PXB bridges on a RunPod container: ncclCommInitAll 0.35 s, first
// ncclAllReduce never completed).
//
// Capability flags cannot be trusted on such machines, so the only reliable
// test is behavioural: run one small AllReduce end to end, bounded by a
// deadline, and check the numbers that come back. The probe uses its own
// non-blocking streams and its own NCCL communicators so a wedge never
// poisons the real per-rank compute streams, and a wedged probe is recovered
// with ncclCommAbort (which forces the spin kernels to exit). On failure the
// caller reroutes the backend's collective selection to the pinned-host-memory
// AllReduce pipeline ("internal"), which does not depend on P2P at all.
//
// NCCL is reached through dlopen rather than a link-time dependency so this
// file builds identically whether or not the NCCL SDK is installed; when the
// library is absent the probe reports "skipped" and the backend's own
// NCCL-less fallback chain applies anyway.
//
// Two probes live here, run in this order by the caller:
//
//   1. tp_probe_cuda_peer_access - does peer traffic actually move? A tiny
//      cudaMemcpyPeer between every ordered device pair, verified by reading
//      the bytes back. It needs no NCCL, so it can run BEFORE the process has
//      any communicator and its verdict can still change how the first one is
//      built. When it says "advertised but broken", the caller disables P2P for
//      NCCL (NCCL_P2P_DISABLE=1) and NCCL falls back to its shared-memory
//      transport, which works for any rank count.
//   2. tp_probe_cuda_collective - the end-to-end NCCL AllReduce check, run with
//      whatever transport step 1 left configured, as the final gate before the
//      backend's own communicator is created.
//
// Ordering matters: NCCL caches its tunables (NCCL_P2P_DISABLE among them) on
// the first communicator init in the process, so a "disable P2P and retry"
// decision has to be made before any NCCL init - including the probe's own.
//
// Verdicts are cached on disk keyed by driver/NCCL versions, the PCI bus ids of
// the probed devices and the probe kind, so a healthy host pays the probes once
// and a broken host pays the timeout once, not on every model load.
//
//   TS_GGML_TP_AR_PROBE=0       skip the probes entirely
//   TS_GGML_TP_AR_PROBE=force   re-probe, ignoring the cached verdicts
//   TS_GGML_TP_AR_PROBE_MS=N    collective completion deadline (default 10000)

// CUDA quant declarations must precede ggml_ops_internal.h's CPU declarations.
#include "ggml-cuda/common.cuh"
#include "ggml_ops_internal.h"
#include "ggml_ops_tp_collective.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <sys/stat.h>
#endif

namespace tsg
{
namespace
{
#if !defined(_WIN32)

    // Minimal NCCL ABI surface. The enum values are fixed across every NCCL 2.x
    // release, and declaring them here avoids a build-time dependency on the
    // NCCL headers (the library itself is loaded with dlopen).
    typedef void* probe_nccl_comm_t;
    constexpr int k_nccl_success = 0;
    constexpr int k_nccl_float32 = 7;
    constexpr int k_nccl_sum = 0;

    struct NcclApi
    {
        void* handle = nullptr;
        int (*comm_init_all)(probe_nccl_comm_t*, int, const int*) = nullptr;
        int (*all_reduce)(const void*, void*, size_t, int, int, probe_nccl_comm_t, cudaStream_t) = nullptr;
        int (*all_gather)(const void*, void*, size_t, int, probe_nccl_comm_t, cudaStream_t) = nullptr;
        int (*group_start)() = nullptr;
        int (*group_end)() = nullptr;
        int (*comm_abort)(probe_nccl_comm_t) = nullptr;
        int (*comm_destroy)(probe_nccl_comm_t) = nullptr;
        int (*get_version)(int*) = nullptr;

        bool load()
        {
            handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
            if (handle == nullptr)
                handle = dlopen("libnccl.so", RTLD_NOW | RTLD_LOCAL);
            if (handle == nullptr)
                return false;
            comm_init_all = reinterpret_cast<int (*)(probe_nccl_comm_t*, int, const int*)>(dlsym(handle, "ncclCommInitAll"));
            all_reduce = reinterpret_cast<int (*)(const void*, void*, size_t, int, int, probe_nccl_comm_t, cudaStream_t)>(dlsym(handle, "ncclAllReduce"));
            all_gather = reinterpret_cast<int (*)(const void*, void*, size_t, int, probe_nccl_comm_t, cudaStream_t)>(dlsym(handle, "ncclAllGather"));
            group_start = reinterpret_cast<int (*)()>(dlsym(handle, "ncclGroupStart"));
            group_end = reinterpret_cast<int (*)()>(dlsym(handle, "ncclGroupEnd"));
            comm_abort = reinterpret_cast<int (*)(probe_nccl_comm_t)>(dlsym(handle, "ncclCommAbort"));
            comm_destroy = reinterpret_cast<int (*)(probe_nccl_comm_t)>(dlsym(handle, "ncclCommDestroy"));
            get_version = reinterpret_cast<int (*)(int*)>(dlsym(handle, "ncclGetVersion"));
            return comm_init_all != nullptr && all_reduce != nullptr
                && group_start != nullptr && group_end != nullptr
                && comm_abort != nullptr && comm_destroy != nullptr;
        }
    };

    // The backend's default communicator narrows large F32 inputs to BF16.
    // Precision-sensitive plans use this separate communicator on the same
    // ordered compute streams. NCCL remains an optional runtime dependency.
    struct ExactF32Comm
    {
        NcclApi api;
        std::vector<probe_nccl_comm_t> comms;
        std::vector<ggml_backend_t> backends;
        bool attempted = false;
        bool failed = false;

        void clear()
        {
            if (failed && api.comm_abort)
                for (auto comm : comms) if (comm) api.comm_abort(comm);
            // Backends are still alive here: tp_comm_free precedes device teardown.
            for (auto backend : backends) ggml_backend_synchronize(backend);
            if (!failed && api.comm_destroy)
                for (auto comm : comms) if (comm) api.comm_destroy(comm);
            if (api.handle) dlclose(api.handle);
            api = {};
            comms.clear();
            backends.clear();
            attempted = failed = false;
        }
    };
    ExactF32Comm g_exact_f32;

    bool exact_f32_enabled(ggml_backend_t * backends, int count)
    {
        const char * enabled = std::getenv("TS_GGML_TP_F32_NCCL");
        const char * transport = std::getenv("GGML_CUDA_ALLREDUCE");
        if ((enabled && std::strcmp(enabled, "0") == 0)
            || (transport && std::strcmp(transport, "nccl") != 0)) return false;
        if (!backends || count < 2 || count > TSG_MAX_DEVICES) return false;
        for (int r = 0; r < count; ++r)
        {
            if (!backends[r] || !ggml_backend_is_cuda(backends[r])) return false;
            const int device = static_cast<ggml_backend_cuda_context *>(backends[r]->context)->device;
            for (int previous = 0; previous < r; ++previous)
                if (static_cast<ggml_backend_cuda_context *>(backends[previous]->context)->device == device)
                    return false;
        }
        return true;
    }

    bool exact_f32_initialize(ExactF32Comm & state, ggml_backend_t * backends, int count)
    {
        state.attempted = true;
        if (!state.api.load()) return false;
        std::vector<int> devices;
        for (int r = 0; r < count; ++r)
            devices.push_back(static_cast<ggml_backend_cuda_context *>(backends[r]->context)->device);
        state.comms.resize(count, nullptr);
        if (state.api.comm_init_all(state.comms.data(), count, devices.data()) != k_nccl_success)
        {
            // Initialization has not touched model inputs, so falling back is safe.
            state.failed = true;
            state.clear();
            state.attempted = true;
            return false;
        }
        state.backends.assign(backends, backends + count);
        return true;
    }

    void exact_f32_abort(ExactF32Comm & state)
    {
        state.failed = true;
        for (auto & comm : state.comms)
            if (comm) { state.api.comm_abort(comm); comm = nullptr; }
    }

    int exact_f32_submit(ExactF32Comm & state, ggml_tensor * const * tensors, size_t elements)
    {
        int result = state.api.group_start();
        if (result == k_nccl_success)
        {
            for (size_t r = 0; r < state.backends.size(); ++r)
            {
                auto * context = static_cast<ggml_backend_cuda_context *>(state.backends[r]->context);
                const auto status = cudaSetDevice(context->device);
                if (status != cudaSuccess) { result = -static_cast<int>(status); break; }
                const int submitted = state.api.all_reduce(tensors[r]->data, tensors[r]->data,
                    elements, k_nccl_float32, k_nccl_sum, state.comms[r], context->stream());
                if (submitted != k_nccl_success) { result = submitted; break; }
            }
            const int ended = state.api.group_end();
            if (result == k_nccl_success) result = ended;
        }
        if (result != k_nccl_success) exact_f32_abort(state);
        return result;
    }

    int probe_timeout_ms()
    {
        const char* value = std::getenv("TS_GGML_TP_AR_PROBE_MS");
        if (value == nullptr || value[0] == '\0')
            return 10000;
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed > 0 ? static_cast<int>(parsed) : 0;
    }

    // Cache key: anything that could change the P2P behaviour. Driver and NCCL
    // versions cover host software updates; the PCI bus ids cover device
    // reassignment (a re-provisioned cloud instance gets fresh ids). `kind`
    // separates the peer-copy verdict from the collective verdict so the two
    // probes do not overwrite each other.
    std::string probe_cache_key(const NcclApi& api, const int* device_indices, int count,
                                const char* kind)
    {
        int driver_version = 0;
        (void)cudaDriverGetVersion(&driver_version);
        int nccl_version = 0;
        if (api.get_version != nullptr)
            (void)api.get_version(&nccl_version);

        std::string key = std::string("kind=") + (kind != nullptr ? kind : "collective")
            + " driver=" + std::to_string(driver_version)
            + " nccl=" + std::to_string(nccl_version) + " devs=";
        for (int r = 0; r < count; ++r)
        {
            char bus_id[64] = {};
            if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device_indices[r]) != cudaSuccess)
                std::snprintf(bus_id, sizeof(bus_id), "idx%d", device_indices[r]);
            key += bus_id;
            key += (r + 1 < count) ? "," : "";
        }
        return key;
    }

    std::string probe_cache_path()
    {
        const char* xdg = std::getenv("XDG_CACHE_HOME");
        std::string dir;
        if (xdg != nullptr && xdg[0] != '\0')
            dir = xdg;
        else
        {
            const char* home = std::getenv("HOME");
            if (home == nullptr || home[0] == '\0')
                return std::string();
            dir = std::string(home) + "/.cache";
        }
        dir += "/tensorsharp";
        ::mkdir(dir.c_str(), 0755);
        return dir + "/tp-collective-probe";
    }

    // The cache file holds one `<key>\t<verdict>` line per probe kind. (Older
    // builds wrote a single key/verdict pair on two lines; such a file simply
    // matches nothing here and is rewritten in the new format.)
    std::vector<std::pair<std::string, std::string>> probe_cache_load(const std::string& path)
    {
        std::vector<std::pair<std::string, std::string>> entries;
        if (path.empty())
            return entries;
        std::ifstream in(path);
        if (!in.is_open())
            return entries;
        std::string line;
        while (std::getline(in, line))
        {
            const std::size_t tab = line.find('\t');
            if (tab == std::string::npos)
                continue;
            entries.emplace_back(line.substr(0, tab), line.substr(tab + 1));
        }
        return entries;
    }

    // 1 = cached ok, 0 = cached broken, -1 = no usable cache entry.
    int probe_cache_read(const std::string& path, const std::string& key)
    {
        for (const auto& e : probe_cache_load(path))
        {
            if (e.first != key)
                continue;
            if (e.second == "ok")
                return 1;
            if (e.second == "broken")
                return 0;
            return -1;
        }
        return -1;
    }

    void probe_cache_write(const std::string& path, const std::string& key, bool ok)
    {
        if (path.empty())
            return;
        auto entries = probe_cache_load(path);
        bool replaced = false;
        for (auto& e : entries)
        {
            if (e.first == key)
            {
                e.second = ok ? "ok" : "broken";
                replaced = true;
                break;
            }
        }
        if (!replaced)
            entries.emplace_back(key, ok ? "ok" : "broken");

        const std::string tmp = path + ".tmp";
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out.is_open())
                return;
            for (const auto& e : entries)
                out << e.first << "\t" << e.second << "\n";
        }
        std::rename(tmp.c_str(), path.c_str());
    }

    struct ProbeState
    {
        // Written by the worker; read by the outer thread after join/deadline.
        // 1 = collective verified, 0 = collective wedged/corrupt (reroute and
        // cache), -1 = inconclusive (probe setup failed, or NCCL's init itself
        // errored — ggml's own init chain reports and handles that case, so
        // no reroute and no cached verdict).
        std::atomic<int> verdict{ 0 };
        std::atomic<bool> finished{ false };
    };

    // The actual end-to-end collective test. Runs on a dedicated thread so a
    // hang anywhere inside NCCL (communicator init included) can be bounded by
    // the caller instead of hanging the model load. The state is shared-owned
    // because an abandoned (detached) worker may outlive the caller's frame.
    void probe_worker(const NcclApi api, std::vector<int> dev_ids, int timeout_ms,
        std::shared_ptr<ProbeState> state)
    {
        const int n = static_cast<int>(dev_ids.size());
        constexpr int k_count = 4096;
        const size_t bytes = k_count * sizeof(float);
        // Expected post-reduction value in every element: sum of (r+1) over ranks.
        const float expected = static_cast<float>(n * (n + 1) / 2);

        std::vector<cudaStream_t> streams(n, nullptr);
        std::vector<float*> bufs(n, nullptr);
        std::vector<probe_nccl_comm_t> comms(n, nullptr);
        bool comms_live = false;
        bool streams_idle = true;

        auto fail = [&](const char* stage, int rc, int verdict)
        {
            std::fprintf(stderr,
                "[TP] collective probe: %s failed (rc=%d)%s\n",
                stage, rc,
                verdict == 0 ? "; treating the device collective as unusable."
                             : "; probe inconclusive, leaving the collective selection alone.");
            std::fflush(stderr);
            state->verdict.store(verdict, std::memory_order_release);
        };

        do
        {
            bool setup_ok = true;
            std::vector<float> host(k_count);
            for (int r = 0; r < n && setup_ok; ++r)
            {
                cudaError_t err = cudaSetDevice(dev_ids[r]);
                if (err == cudaSuccess)
                    err = cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking);
                if (err == cudaSuccess)
                    err = cudaMalloc(&bufs[r], bytes);
                if (err == cudaSuccess)
                {
                    std::fill(host.begin(), host.end(), static_cast<float>(r + 1));
                    err = cudaMemcpyAsync(bufs[r], host.data(), bytes, cudaMemcpyHostToDevice, streams[r]);
                }
                if (err == cudaSuccess)
                    err = cudaStreamSynchronize(streams[r]);
                if (err != cudaSuccess)
                {
                    // Not evidence about the collective — do not reroute or cache.
                    fail("device setup", static_cast<int>(err), -1);
                    setup_ok = false;
                }
            }
            if (!setup_ok)
                break;

            int rc = api.comm_init_all(comms.data(), n, dev_ids.data());
            if (rc != k_nccl_success)
            {
                // An init *error* (as opposed to a hang) is handled by ggml's own
                // init chain, which falls back to the internal pipeline itself.
                fail("ncclCommInitAll", rc, -1);
                break;
            }
            comms_live = true;

            rc = api.group_start();
            for (int r = 0; r < n && rc == k_nccl_success; ++r)
            {
                cudaSetDevice(dev_ids[r]);
                rc = api.all_reduce(bufs[r], bufs[r], k_count, k_nccl_float32, k_nccl_sum, comms[r], streams[r]);
            }
            if (rc == k_nccl_success)
                rc = api.group_end();
            if (rc != k_nccl_success)
            {
                // ggml's per-call NCCL_CHECK would abort the process on the same
                // error mid-inference, so a deterministic enqueue failure is a
                // reason to reroute.
                fail("ncclAllReduce enqueue", rc, 0);
                break;
            }

            // Bounded completion wait. cudaStreamQuery never blocks, so a
            // wedged collective is detected by the deadline instead of hanging
            // here the way cudaStreamSynchronize would.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
            bool done = false;
            while (!done)
            {
                done = true;
                for (int r = 0; r < n; ++r)
                {
                    cudaSetDevice(dev_ids[r]);
                    if (cudaStreamQuery(streams[r]) != cudaSuccess)
                    {
                        done = false;
                        break;
                    }
                }
                if (done)
                    break;
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    std::fprintf(stderr,
                        "[TP] collective probe: AllReduce did not complete within %d ms — the driver advertises "
                        "P2P but peer traffic never arrives on this host. Aborting the probe communicators.\n",
                        timeout_ms);
                    std::fflush(stderr);
                    state->verdict.store(0, std::memory_order_release);
                    streams_idle = false;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!done)
                break;

            // Completion is not enough: broken peer paths have been observed to
            // deliver garbage rather than hang, so check the arithmetic too.
            bool values_ok = true;
            for (int r = 0; r < n && values_ok; ++r)
            {
                cudaSetDevice(dev_ids[r]);
                if (cudaMemcpy(host.data(), bufs[r], bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                {
                    values_ok = false;
                    break;
                }
                for (int i = 0; i < k_count; ++i)
                {
                    if (host[i] != expected)
                    {
                        values_ok = false;
                        break;
                    }
                }
            }
            if (!values_ok)
            {
                std::fprintf(stderr,
                    "[TP] collective probe: AllReduce completed but returned wrong sums — "
                    "peer transfers are silently corrupt on this host.\n");
                std::fflush(stderr);
                state->verdict.store(0, std::memory_order_release);
                break;
            }

            state->verdict.store(1, std::memory_order_release);
        } while (false);

        // Teardown. A wedged collective is recovered with ncclCommAbort, which
        // sets the communicator's abort flag and makes the spinning kernels
        // exit; the healthy path uses the ordinary destroy.
        if (comms_live)
        {
            const bool wedged = !streams_idle;
            for (int r = 0; r < n; ++r)
            {
                if (comms[r] == nullptr)
                    continue;
                if (wedged)
                    api.comm_abort(comms[r]);
                else
                    api.comm_destroy(comms[r]);
            }
            if (wedged)
            {
                // Give the aborted kernels a moment to drain so the buffers and
                // streams can be released instead of leaked.
                const auto drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
                streams_idle = true;
                for (int r = 0; r < n; ++r)
                {
                    cudaSetDevice(dev_ids[r]);
                    while (cudaStreamQuery(streams[r]) != cudaSuccess)
                    {
                        if (std::chrono::steady_clock::now() >= drain_deadline)
                        {
                            streams_idle = false;
                            break;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    if (!streams_idle)
                        break;
                }
            }
        }
        if (streams_idle)
        {
            for (int r = 0; r < n; ++r)
            {
                cudaSetDevice(dev_ids[r]);
                if (bufs[r] != nullptr)
                    cudaFree(bufs[r]);
                if (streams[r] != nullptr)
                    cudaStreamDestroy(streams[r]);
            }
        }
        // else: deliberately leak the probe buffers/streams — the device still
        // has kernels wedged on them, and freeing memory a kernel may touch is
        // worse than a few KB lost on a host that is already being demoted to
        // the host-staged collective.

        state->finished.store(true, std::memory_order_release);
    }

    // --- Peer-copy probe -----------------------------------------------------
    // Does a peer copy actually deliver bytes? Every ordered device pair writes
    // a known pattern from src into dst's buffer and dst reads it back. A host
    // that lies about P2P either never completes the copy (covered by the
    // caller's deadline) or leaves dst unchanged, and both are caught here
    // without involving NCCL — which matters because this verdict has to be
    // known before the process creates its first communicator.
    void peer_probe_worker(std::vector<int> dev_ids, std::shared_ptr<ProbeState> state)
    {
        const int n = static_cast<int>(dev_ids.size());
        const std::size_t bytes = 4096;
        const std::size_t words = bytes / sizeof(std::uint32_t);

        std::vector<void*> bufs(static_cast<std::size_t>(n), nullptr);
        std::vector<std::uint32_t> host(words, 0);
        bool ok = true;
        bool setup_failed = false;

        for (int r = 0; r < n && ok; ++r)
        {
            if (cudaSetDevice(dev_ids[r]) != cudaSuccess ||
                cudaMalloc(&bufs[r], bytes) != cudaSuccess)
            {
                // Could not even set the probe up: that says nothing about peer
                // access, so report "inconclusive" rather than convicting the
                // host (or, worse, clearing it).
                ok = false;
                setup_failed = true;
            }
        }
        if (setup_failed)
        {
            for (int r = 0; r < n; ++r)
                if (bufs[r] != nullptr) { cudaSetDevice(dev_ids[r]); cudaFree(bufs[r]); }
            (void)cudaGetLastError();
            state->verdict.store(-1, std::memory_order_release);
            state->finished.store(true, std::memory_order_release);
            return;
        }

        // Enable peer access where the driver claims it is possible. A device
        // pair without peer access is not a failure: NCCL would route it over
        // its own transport, and this probe only judges pairs that claim P2P.
        // Pairs we turn on ourselves are remembered so they can be turned back
        // off before returning: the probe must leave device state exactly as it
        // found it (see the disable loop at the end).
        std::vector<std::pair<int, int>> enabled_by_probe;
        int enabled_pairs = 0;
        for (int i = 0; i < n && ok; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (i == j)
                    continue;
                int can = 0;
                if (cudaDeviceCanAccessPeer(&can, dev_ids[i], dev_ids[j]) != cudaSuccess || can == 0)
                    continue;
                cudaSetDevice(dev_ids[i]);
                const cudaError_t err = cudaDeviceEnablePeerAccess(dev_ids[j], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
                    continue;
                if (err == cudaSuccess)
                    enabled_by_probe.emplace_back(i, j);
                ++enabled_pairs;
            }
        }
        if (enabled_pairs == 0)
        {
            // Nothing claims P2P, so there is no false advertisement to catch.
            std::fprintf(stderr,
                "[TP] peer-copy probe: no device pair advertises peer access; "
                "leaving the transport choice to NCCL.\n");
            std::fflush(stderr);
            state->verdict.store(1, std::memory_order_release);
            state->finished.store(true, std::memory_order_release);
            for (int r = 0; r < n; ++r)
                if (bufs[r] != nullptr) { cudaSetDevice(dev_ids[r]); cudaFree(bufs[r]); }
            return;
        }

        for (int i = 0; i < n && ok; ++i)
        {
            for (int j = 0; j < n && ok; ++j)
            {
                if (i == j)
                    continue;
                int can = 0;
                if (cudaDeviceCanAccessPeer(&can, dev_ids[i], dev_ids[j]) != cudaSuccess || can == 0)
                    continue;

                // Pattern is pair-specific so a stale buffer cannot pass.
                const std::uint32_t pattern = 0xA5A50000u | static_cast<std::uint32_t>(i * 16 + j);
                std::fill(host.begin(), host.end(), pattern);

                cudaSetDevice(dev_ids[i]);
                if (cudaMemcpy(bufs[i], host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) { ok = false; break; }
                cudaSetDevice(dev_ids[j]);
                if (cudaMemset(bufs[j], 0, bytes) != cudaSuccess) { ok = false; break; }
                if (cudaDeviceSynchronize() != cudaSuccess) { ok = false; break; }

                if (cudaMemcpyPeer(bufs[j], dev_ids[j], bufs[i], dev_ids[i], bytes) != cudaSuccess) { ok = false; break; }
                cudaSetDevice(dev_ids[j]);
                if (cudaDeviceSynchronize() != cudaSuccess) { ok = false; break; }

                std::vector<std::uint32_t> back(words, 0);
                if (cudaMemcpy(back.data(), bufs[j], bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { ok = false; break; }
                for (std::size_t w = 0; w < words; ++w)
                {
                    if (back[w] != pattern)
                    {
                        std::fprintf(stderr,
                            "[TP] peer-copy probe: device %d -> %d reported success but delivered "
                            "0x%08X instead of 0x%08X.\n", dev_ids[i], dev_ids[j], back[w], pattern);
                        std::fflush(stderr);
                        ok = false;
                        break;
                    }
                }
            }
        }

        if (ok)
        {
            std::fprintf(stderr,
                "[TP] peer-copy probe: %d advertised device pair(s) verified.\n", enabled_pairs);
            std::fflush(stderr);
        }

        // Hand the devices back exactly as they were found. Peer access is not
        // a free capability to leave switched on: enabling it changes which
        // path the backend's own AllReduce takes, and on this class of host
        // that path is the slow one — a probe that "helpfully" left it enabled
        // cost 30% of tp2 decode throughput.
        for (const auto& pair : enabled_by_probe)
        {
            cudaSetDevice(dev_ids[pair.first]);
            (void)cudaDeviceDisablePeerAccess(dev_ids[pair.second]);
        }

        for (int r = 0; r < n; ++r)
        {
            if (bufs[r] == nullptr)
                continue;
            cudaSetDevice(dev_ids[r]);
            cudaFree(bufs[r]);
        }
        (void)cudaGetLastError();

        state->verdict.store(ok ? 1 : 0, std::memory_order_release);
        state->finished.store(true, std::memory_order_release);
    }

#endif // !defined(_WIN32)
} // namespace

// Behavioural pre-flight of CUDA peer copies on the given devices. Returns 1
// when every pair that advertises P2P actually delivers the bytes, 0 when the
// advertisement is a lie (caller should disable P2P for NCCL before any
// communicator exists), and -1 when the probe does not apply.
int tp_probe_cuda_peer_access(const int* device_indices, int count)
{
#if defined(_WIN32)
    (void)device_indices;
    (void)count;
    return -1;
#else
    if (device_indices == nullptr || count < 2)
        return -1;

    const char* mode = std::getenv("TS_GGML_TP_AR_PROBE");
    if (mode != nullptr && std::strcmp(mode, "0") == 0)
        return -1;
    const bool force = mode != nullptr && std::strcmp(mode, "force") == 0;

    const int timeout_ms = probe_timeout_ms();
    if (timeout_ms <= 0)
        return -1;

    NcclApi api;                       // only for the cache key's NCCL version
    api.load();

    const std::string cache_path = probe_cache_path();
    const std::string cache_key = probe_cache_key(api, device_indices, count, "peer");
    if (!force)
    {
        const int cached = probe_cache_read(cache_path, cache_key);
        if (cached >= 0)
            return cached;
    }

    auto state = std::make_shared<ProbeState>();
    std::vector<int> dev_ids(device_indices, device_indices + count);
    std::thread worker(peer_probe_worker, dev_ids, state);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!state->finished.load(std::memory_order_acquire)
        && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!state->finished.load(std::memory_order_acquire))
    {
        // A peer copy that never returns is the same verdict as one that
        // delivers garbage, and the thread cannot be joined — leave it.
        worker.detach();
        std::fprintf(stderr,
            "[TP] peer-copy probe: a peer copy did not complete within %d ms; "
            "treating peer access as non-functional on this host.\n", timeout_ms);
        std::fflush(stderr);
        probe_cache_write(cache_path, cache_key, false);
        return 0;
    }

    worker.join();
    const int verdict = state->verdict.load(std::memory_order_acquire);
    if (verdict >= 0)                 // never cache an inconclusive probe
        probe_cache_write(cache_path, cache_key, verdict == 1);
    return verdict;
#endif
}

// Behavioural pre-flight of the NCCL collective on the given CUDA devices.
// Returns 1 when a small AllReduce completes with correct sums, 0 when it
// hangs / errors / corrupts (caller should force the internal AllReduce), and
// -1 when the probe does not apply (Windows, NCCL absent, probe disabled).
int tp_probe_cuda_collective(const int* device_indices, int count)
{
#if defined(_WIN32)
    (void)device_indices;
    (void)count;
    return -1;
#else
    if (device_indices == nullptr || count < 2)
        return -1;

    const char* mode = std::getenv("TS_GGML_TP_AR_PROBE");
    if (mode != nullptr && std::strcmp(mode, "0") == 0)
        return -1;
    const bool force = mode != nullptr && std::strcmp(mode, "force") == 0;

    const int timeout_ms = probe_timeout_ms();
    if (timeout_ms <= 0)
        return -1;

    NcclApi api;
    if (!api.load())
        return -1; // No NCCL in the process: the backend cannot pick it either.

    const std::string cache_path = probe_cache_path();
    // The collective's behaviour depends on the transport it is allowed to use,
    // so a verdict reached with P2P disabled must not be reused when P2P is on.
    const char* p2p_off = std::getenv("NCCL_P2P_DISABLE");
    const std::string cache_key = probe_cache_key(
        api, device_indices, count,
        (p2p_off != nullptr && p2p_off[0] == '1') ? "collective-shm" : "collective");
    if (!force)
    {
        const int cached = probe_cache_read(cache_path, cache_key);
        if (cached == 0)
        {
            std::fprintf(stderr,
                "[TP] collective probe: cached verdict is 'broken' for this host (%s); "
                "set TS_GGML_TP_AR_PROBE=force to re-test.\n",
                cache_path.c_str());
            std::fflush(stderr);
            return 0;
        }
        if (cached == 1)
            return 1;
    }

    auto state = std::make_shared<ProbeState>();
    std::vector<int> dev_ids(device_indices, device_indices + count);
    std::thread worker(probe_worker, api, dev_ids, timeout_ms, state);

    // Outer safety net: the in-thread deadline covers a wedged collective, but
    // ncclCommInitAll itself can hang on a sufficiently broken host. Give the
    // worker the collective deadline plus generous setup time; if it still has
    // not finished, abandon it (detach) and report the collective unusable.
    const auto outer_deadline = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(timeout_ms + 20000);
    while (!state->finished.load(std::memory_order_acquire)
        && std::chrono::steady_clock::now() < outer_deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    int verdict;
    if (state->finished.load(std::memory_order_acquire))
    {
        worker.join();
        verdict = state->verdict.load(std::memory_order_acquire);
    }
    else
    {
        std::fprintf(stderr,
            "[TP] collective probe: probe thread did not finish (NCCL init hang?); "
            "abandoning it and treating the device collective as unusable.\n");
        std::fflush(stderr);
        worker.detach();
        verdict = 0;
    }

    // Only definitive verdicts are cached: an inconclusive probe (setup or
    // NCCL-init error) must be allowed to retry on the next load.
    if (verdict == 0 || verdict == 1)
        probe_cache_write(cache_path, cache_key, verdict == 1);
    return verdict;
#endif // defined(_WIN32)
}

int tp_cuda_allreduce_f32(ggml_backend_t* backends, ggml_tensor** tensors, int count)
{
#if defined(_WIN32)
    (void)backends; (void)tensors; (void)count;
    return 0;
#else
    if (!exact_f32_enabled(backends, count)) return 0;
    const int64_t elements = ggml_nelements(tensors[0]);
    if (elements == 0) return 1;
    auto& state = g_exact_f32;
    if (!state.attempted)
    {
        if (!exact_f32_initialize(state, backends, count)) return 0;
        std::fprintf(stderr, "[TP] TensorSharp F32 NCCL transport: %d ranks, no BF16 compression.\n", count);
    }
    if (state.comms.empty()) return 0;
    if (state.failed || state.backends.size() != static_cast<size_t>(count)
        || !std::equal(state.backends.begin(), state.backends.end(), backends))
    {
        set_last_error("Tensor-parallel F32 communicator is failed or belongs to a different device group.");
        return -1;
    }
    // Once a collective has started, an error must abort this forward, never
    // fall back and accidentally sum an already-modified buffer a second time.
    const int result = exact_f32_submit(state, tensors, static_cast<size_t>(elements));
    if (result != k_nccl_success)
    {
        set_last_error("Tensor-parallel F32 NCCL collective failed: " + std::to_string(result));
        return -1;
    }
    return 1;
#endif
}

void tp_cuda_allreduce_f32_free()
{
#if !defined(_WIN32)
    g_exact_f32.clear();
#endif
}

struct TpF32Gather
{
#if !defined(_WIN32)
    ExactF32Comm state;
    std::vector<void *> scratch;
    size_t scratch_bytes = 0;
    ~TpF32Gather()
    {
        // Drain/abort collectives before freeing their staging allocations.
        int devices[TSG_MAX_DEVICES] = {};
        for (size_t r = 0; r < scratch.size(); ++r)
            devices[r] = static_cast<ggml_backend_cuda_context *>(state.backends[r]->context)->device;
        state.clear();
        for (size_t r = 0; r < scratch.size(); ++r)
            if (scratch[r])
            {
                cudaSetDevice(devices[r]);
                cudaFree(scratch[r]);
            }
    }
#endif
};

TpF32Gather * tp_cuda_f32_gather_create(ggml_backend_t * backends, int count)
{
#if defined(_WIN32)
    (void)backends; (void)count;
    return nullptr;
#else
    if (!exact_f32_enabled(backends, count)) return nullptr;
    // This executor owns backends outside the generic TP device group, so run
    // the same cached behavioural checks before its first communicator exists.
    if (std::getenv("GGML_CUDA_ALLREDUCE") == nullptr)
    {
        std::vector<int> devices;
        for (int r = 0; r < count; ++r)
            devices.push_back(static_cast<ggml_backend_cuda_context *>(backends[r]->context)->device);
        if (std::getenv("NCCL_P2P_DISABLE") == nullptr
            && tp_probe_cuda_peer_access(devices.data(), count) == 0)
        {
            setenv("NCCL_P2P_DISABLE", "1", 0);
            std::fprintf(stderr, "[TP] F32 gather: using NCCL shared-memory transport after failed peer-copy probe.\n");
        }
        if (tp_probe_cuda_collective(devices.data(), count) == 0) return nullptr;
    }
    auto handle = std::make_unique<TpF32Gather>();
    if (!exact_f32_initialize(handle->state, backends, count)) return nullptr;
    if (!handle->state.api.all_gather) return nullptr;
    handle->scratch.resize(count, nullptr);
    std::fprintf(stderr, "[TP] TensorSharp F32 NCCL row gather: %d ranks.\n", count);
    return handle.release();
#endif
}

bool tp_cuda_f32_gather(TpF32Gather * handle,
    ggml_tensor * const * sources, ggml_tensor * const * destinations,
    const int64_t * first_rows, const int64_t * row_counts,
    int64_t full_rows, int64_t columns, std::string & error)
{
    error.clear();
#if defined(_WIN32)
    (void)handle; (void)sources; (void)destinations; (void)first_rows;
    (void)row_counts; (void)full_rows; (void)columns;
    error = "F32 CUDA gather is unavailable on this platform.";
    return false;
#else
    if (!handle || handle->state.failed || handle->state.comms.empty())
    {
        error = "F32 CUDA gather has no healthy communicator.";
        return false;
    }
    if (!sources || !destinations || !first_rows || !row_counts || full_rows <= 0 || columns <= 0
        || uint64_t(full_rows) > std::numeric_limits<size_t>::max() / sizeof(float) / uint64_t(columns)
        || full_rows > std::numeric_limits<int64_t>::max() / columns)
    {
        error = "F32 CUDA gather has invalid dimensions or arrays.";
        return false;
    }
    auto & state = handle->state;
    const auto count = state.backends.size();
    int64_t covered = 0, max_rows = 0;
    for (size_t r = 0; r < count; ++r)
    {
        auto * src = sources[r];
        auto * dst = destinations[r];
        if (first_rows[r] != covered || row_counts[r] <= 0 || row_counts[r] > full_rows - covered
            || !src || !dst || !src->data || !dst->data
            || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32
            || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)
            || src->ne[0] != row_counts[r] || dst->ne[0] != full_rows
            || ggml_nelements(src) != row_counts[r] * columns
            || ggml_nelements(dst) != full_rows * columns)
        {
            error = "F32 CUDA gather requires a complete ordered row tiling and contiguous F32 tensors.";
            return false;
        }
        const auto source_address = reinterpret_cast<uintptr_t>(src->data);
        const auto destination_address = reinterpret_cast<uintptr_t>(dst->data);
        const auto source_bytes = static_cast<size_t>(row_counts[r] * columns) * sizeof(float);
        const auto destination_bytes = static_cast<size_t>(full_rows * columns) * sizeof(float);
        if ((source_address <= destination_address && destination_address - source_address < source_bytes)
            || (destination_address < source_address && source_address - destination_address < destination_bytes))
        {
            error = "F32 CUDA gather source and destination must not overlap.";
            return false;
        }
        auto * source_buffer = src->buffer ? src->buffer : (src->view_src ? src->view_src->buffer : nullptr);
        auto * destination_buffer = dst->buffer ? dst->buffer : (dst->view_src ? dst->view_src->buffer : nullptr);
        const auto device = ggml_backend_get_device(state.backends[r]);
        if (!source_buffer || !destination_buffer
            || ggml_backend_buft_get_device(ggml_backend_buffer_get_type(source_buffer)) != device
            || ggml_backend_buft_get_device(ggml_backend_buffer_get_type(destination_buffer)) != device)
        {
            error = "F32 CUDA gather tensors must reside on their rank's CUDA device.";
            return false;
        }
        covered += row_counts[r];
        max_rows = std::max(max_rows, row_counts[r]);
    }
    if (covered != full_rows)
    {
        error = "F32 CUDA gather row tiling is incomplete.";
        return false;
    }
    // CUDA allocation addresses are unique across the participating devices.
    // Reject cross-rank aliases before modifying any destination.
    for (size_t r = 0; r < count; ++r)
        for (size_t other = 0; other < count; ++other)
        {
            const auto dst = reinterpret_cast<uintptr_t>(destinations[r]->data);
            const auto src = reinterpret_cast<uintptr_t>(sources[other]->data);
            const auto out = reinterpret_cast<uintptr_t>(destinations[other]->data);
            const auto output_bytes = static_cast<size_t>(full_rows * columns) * sizeof(float);
            const auto input_bytes = static_cast<size_t>(row_counts[other] * columns) * sizeof(float);
            const bool source_overlap = (src <= dst && dst - src < input_bytes)
                || (dst < src && src - dst < output_bytes);
            const bool output_overlap = r != other && ((out <= dst && dst - out < output_bytes)
                || (dst < out && out - dst < output_bytes));
            if (source_overlap || output_overlap)
            {
                error = "F32 CUDA gather tensors alias another rank's buffers.";
                return false;
            }
        }
    const auto chunk_elements = static_cast<size_t>(max_rows * columns);
    if (chunk_elements > std::numeric_limits<size_t>::max() / sizeof(float) / count)
    {
        error = "F32 CUDA gather staging size overflows.";
        return false;
    }
    const auto chunk_bytes = chunk_elements * sizeof(float);
    const auto staging_bytes = chunk_bytes * count;
    if (handle->scratch_bytes < staging_bytes)
    {
        for (auto backend : state.backends) ggml_backend_synchronize(backend);
        handle->scratch_bytes = 0;
        for (size_t r = 0; r < count; ++r)
        {
            auto * context = static_cast<ggml_backend_cuda_context *>(state.backends[r]->context);
            auto status = cudaSetDevice(context->device);
            if (status == cudaSuccess && handle->scratch[r])
            {
                status = cudaFree(handle->scratch[r]);
                handle->scratch[r] = nullptr;
            }
            if (status == cudaSuccess) status = cudaMalloc(&handle->scratch[r], staging_bytes);
            if (status != cudaSuccess)
            {
                error = std::string("F32 CUDA gather staging allocation failed: ") + cudaGetErrorString(status);
                return false;
            }
        }
        handle->scratch_bytes = staging_bytes;
    }
    // Pack into an in-place AllGather slot. Copying bits, rather than summing
    // zero-padded floats, also preserves signed zero and subnormal values.
    const auto staging_pitch = static_cast<size_t>(max_rows) * sizeof(float);
    const auto output_pitch = static_cast<size_t>(full_rows) * sizeof(float);
    for (size_t r = 0; r < count; ++r)
    {
        auto * context = static_cast<ggml_backend_cuda_context *>(state.backends[r]->context);
        auto * slot = static_cast<char *>(handle->scratch[r]) + r * chunk_bytes;
        auto status = cudaSetDevice(context->device);
        if (status == cudaSuccess)
            status = cudaMemsetAsync(slot, 0, chunk_bytes, context->stream());
        if (status == cudaSuccess)
        {
            const auto width = static_cast<size_t>(row_counts[r]) * sizeof(float);
            status = cudaMemcpy2DAsync(slot, staging_pitch, sources[r]->data, width,
                width, static_cast<size_t>(columns), cudaMemcpyDeviceToDevice, context->stream());
        }
        if (status != cudaSuccess)
        {
            exact_f32_abort(state);
            error = std::string("F32 CUDA gather packing failed: ") + cudaGetErrorString(status);
            return false;
        }
    }
    int result = state.api.group_start();
    if (result == k_nccl_success)
    {
        for (size_t r = 0; r < count; ++r)
        {
            auto * context = static_cast<ggml_backend_cuda_context *>(state.backends[r]->context);
            const auto status = cudaSetDevice(context->device);
            if (status != cudaSuccess) { result = -static_cast<int>(status); break; }
            const auto * slot = static_cast<char *>(handle->scratch[r]) + r * chunk_bytes;
            result = state.api.all_gather(slot, handle->scratch[r], chunk_elements,
                k_nccl_float32, state.comms[r], context->stream());
            if (result != k_nccl_success) break;
        }
        const int ended = state.api.group_end();
        if (result == k_nccl_success) result = ended;
    }
    if (result != k_nccl_success)
    {
        exact_f32_abort(state);
        error = "F32 CUDA gather collective failed: " + std::to_string(result);
        return false;
    }
    for (size_t r = 0; r < count; ++r)
    {
        auto * context = static_cast<ggml_backend_cuda_context *>(state.backends[r]->context);
        auto status = cudaSetDevice(context->device);
        for (size_t source_rank = 0; status == cudaSuccess && source_rank < count; ++source_rank)
        {
            const auto * slot = static_cast<char *>(handle->scratch[r]) + source_rank * chunk_bytes;
            auto * destination = static_cast<float *>(destinations[r]->data) + first_rows[source_rank];
            const auto width = static_cast<size_t>(row_counts[source_rank]) * sizeof(float);
            status = cudaMemcpy2DAsync(destination, output_pitch, slot, staging_pitch,
                width, static_cast<size_t>(columns), cudaMemcpyDeviceToDevice, context->stream());
        }
        if (status != cudaSuccess)
        {
            exact_f32_abort(state);
            error = std::string("F32 CUDA gather unpacking failed: ") + cudaGetErrorString(status);
            return false;
        }
    }
    return true;
#endif
}

void tp_cuda_f32_gather_free(TpF32Gather * handle)
{
    delete handle;
}

// ---------------------------------------------------------------------------
// ggml's internal two-GPU AllReduce on hosts without NCCL (every Windows host).
//
// That pipeline synchronizes the two GPUs INSIDE a kernel: each GPU publishes
// an arrival token in mapped pinned host memory and spins until it reads the
// other GPU's token. Nothing in the CUDA programming model promises that the
// two kernels make progress together, and on WDDM (the Windows driver model)
// a launch can sit in the driver's per-device software queue until something
// flushes that queue. The pipeline launches on GPU 0, then GPU 1, and the
// caller then synchronizes GPU 0: GPU 0's kernel is submitted and spins while
// GPU 1's can wait, unsubmitted, behind a synchronize that never returns. The
// spin sleeps between reads, so CPU and GPU both look idle - the hang reported
// in issue #256 when --layer-split 2 was replaced by --tp 2 on 2x RTX 3080
// under Windows 10.
//
// Two defences, both TensorSharp-side (ggml stays unmodified):
//   * tp_cuda_flush_backends(): after every AllReduce TensorSharp issues,
//     flush each rank's queue (cudaStreamQuery submits pending work without
//     waiting), so both halves of the rendezvous are on their GPUs before any
//     thread blocks.
//   * tp_probe_cuda_host_signal(): before the communicator exists, run the
//     same rendezvous with TensorSharp's own kernels - which give up on their
//     own after a deadline - and fall back to the host reduction when the two
//     GPUs cannot see each other's signal at all.
// ---------------------------------------------------------------------------

namespace
{
    // Mapped pinned layout, in ints: one 64-byte line per value per rank.
    constexpr int k_signal_line_ints = 16;
    constexpr int k_signal_arrival = 0;                         // + rank line
    constexpr int k_signal_payload = 2 * k_signal_line_ints;    // + rank line
    constexpr int k_signal_verdict = 4 * k_signal_line_ints;    // + rank line
    constexpr int k_signal_ints = 6 * k_signal_line_ints;

    // Kernel verdicts written back to host memory.
    constexpr int k_signal_ok = 1;
    constexpr int k_signal_timeout = 2;
    constexpr int k_signal_bad_payload = 3;
    constexpr int k_signal_unsupported = 4;

    __global__ void tp_host_signal_probe_kernel(int* base, int rank, int token, long long timeout_ns)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0)
            return;
        volatile int* verdict = base + k_signal_verdict + rank * k_signal_line_ints;
#if __CUDA_ARCH__ >= 700
        volatile int* arrival_mine = base + k_signal_arrival + rank * k_signal_line_ints;
        volatile int* arrival_peer = base + k_signal_arrival + (1 - rank) * k_signal_line_ints;
        volatile int* payload_mine = base + k_signal_payload + rank * k_signal_line_ints;
        volatile int* payload_peer = base + k_signal_payload + (1 - rank) * k_signal_line_ints;
        // Same ordering as ggml's chunked AllReduce kernel: data, fence, token.
        *payload_mine = token * 2 + rank;
        __threadfence_system();
        *arrival_mine = token;
        __threadfence_system();

        unsigned long long start, now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
        int result = k_signal_timeout;
        for (;;)
        {
            if (*arrival_peer == token)
            {
                __threadfence_system();
                result = *payload_peer == token * 2 + (1 - rank) ? k_signal_ok : k_signal_bad_payload;
                break;
            }
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            if (static_cast<long long>(now - start) > timeout_ns)
                break;
            __nanosleep(200);
        }
        *verdict = result;
        __threadfence_system();
#else
        (void)token;
        (void)timeout_ns;
        *verdict = k_signal_unsupported;
#endif
    }

    // Restores the calling thread's CUDA device on scope exit.
    struct CudaDeviceRestore
    {
        int device = -1;
        CudaDeviceRestore() { if (cudaGetDevice(&device) != cudaSuccess) device = -1; }
        ~CudaDeviceRestore() { if (device >= 0) cudaSetDevice(device); }
    };
}

int tp_probe_cuda_host_signal(const int* device_indices, int count)
{
    if (device_indices == nullptr || count != 2)
        return -1;
    const char* mode = std::getenv("TS_GGML_TP_AR_PROBE");
    if (mode != nullptr && std::strcmp(mode, "0") == 0)
        return -1;
    const int device_count = ggml_cuda_info().device_count;
    for (int r = 0; r < count; ++r)
    {
        // ggml's pipeline declines devices below Volta on its own.
        if (device_indices[r] < 0 || device_indices[r] >= device_count
            || ggml_cuda_info().devices[device_indices[r]].cc < GGML_CUDA_CC_VOLTA)
            return -1;
    }

    // The kernels give up after kernel_ms; the host waits that long plus time for
    // a launch to be scheduled at all. A healthy pair meets in microseconds.
    constexpr int kernel_ms = 3000;
    constexpr int host_grace_ms = 5000;
    constexpr int token = 1;

    CudaDeviceRestore restore;
    // Allocated the way ggml's pipeline allocates its arrival ring: portable,
    // mapped, one device pointer shared by both GPUs.
    ggml_cuda_set_device(device_indices[count - 1]);
    int* host = nullptr;
    int* mapped = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void**>(&host), k_signal_ints * sizeof(int),
            cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess)
    {
        (void)cudaGetLastError();
        return -1;
    }
    std::memset(host, 0, k_signal_ints * sizeof(int));
    if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), host, 0) != cudaSuccess)
    {
        (void)cudaGetLastError();
        cudaFreeHost(host);
        return -1;
    }

    cudaStream_t streams[2] = {};
    bool launched[2] = {};
    for (int r = 0; r < count; ++r)
    {
        ggml_cuda_set_device(device_indices[r]);
        if (cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking) != cudaSuccess)
            break;
        tp_host_signal_probe_kernel<<<1, 32, 0, streams[r]>>>(mapped, r, token,
            static_cast<long long>(kernel_ms) * 1000000LL);
        if (cudaGetLastError() != cudaSuccess)
            break;
        launched[r] = true;
        // Submit now: the rendezvous needs both kernels on their GPUs at once.
        (void)cudaStreamQuery(streams[r]);
    }
    if (!launched[0] || !launched[1])
    {
        // A kernel whose peer never launched gives up on its own after
        // kernel_ms; wait for it before freeing the page it writes.
        for (int r = 0; r < count; ++r)
            if (streams[r] != nullptr)
            {
                ggml_cuda_set_device(device_indices[r]);
                cudaStreamSynchronize(streams[r]);
                cudaStreamDestroy(streams[r]);
            }
        (void)cudaGetLastError();
        cudaFreeHost(host);
        return -1;
    }

    bool done[2] = {};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kernel_ms + host_grace_ms);
    for (;;)
    {
        bool all = true;
        for (int r = 0; r < count; ++r)
        {
            if (done[r])
                continue;
            ggml_cuda_set_device(device_indices[r]);
            const cudaError_t rc = cudaStreamQuery(streams[r]);
            if (rc == cudaSuccess)
                done[r] = true;
            else if (rc != cudaErrorNotReady)
            {
                std::fprintf(stderr, "[TP] pinned-host AllReduce probe: GPU %d failed (%s).\n",
                    device_indices[r], cudaGetErrorString(rc));
                std::fflush(stderr);
                (void)cudaGetLastError();
                return 0;
            }
            all &= done[r];
        }
        if (all || std::chrono::steady_clock::now() >= deadline)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!done[0] || !done[1])
    {
        // A kernel that has not even run by now cannot be cleaned up safely: it
        // would still write into the mapped page. Leave the page and the streams.
        std::fprintf(stderr,
            "[TP] pinned-host AllReduce probe: GPU %d's kernel did not run within %d ms.\n",
            device_indices[done[0] ? 1 : 0], kernel_ms + host_grace_ms);
        std::fflush(stderr);
        return 0;
    }

    int verdicts[2];
    for (int r = 0; r < count; ++r)
        verdicts[r] = reinterpret_cast<volatile int*>(host)[k_signal_verdict + r * k_signal_line_ints];
    for (int r = 0; r < count; ++r)
    {
        ggml_cuda_set_device(device_indices[r]);
        cudaStreamDestroy(streams[r]);
    }
    cudaFreeHost(host);

    if (verdicts[0] == k_signal_ok && verdicts[1] == k_signal_ok)
        return 1;
    if (verdicts[0] == k_signal_unsupported || verdicts[1] == k_signal_unsupported)
        return -1;
    for (int r = 0; r < count; ++r)
        if (verdicts[r] != k_signal_ok)
            std::fprintf(stderr, "[TP] pinned-host AllReduce probe: GPU %d %s GPU %d's signal within %d ms.\n",
                device_indices[r], verdicts[r] == k_signal_bad_payload ? "saw stale data behind" : "never saw",
                device_indices[1 - r], kernel_ms);
    std::fflush(stderr);
    return 0;
}

void tp_cuda_flush_backends(ggml_backend_t const* backends, int count)
{
#if defined(_WIN32)
    // Only WDDM queues launches in software; elsewhere a launch is already on
    // its way to the GPU and this would be a wasted driver call per collective.
    // TS_GGML_TP_WDDM_FLUSH=0 turns it off, for diagnosis only.
    static const bool enabled = []
    {
        const char* value = std::getenv("TS_GGML_TP_WDDM_FLUSH");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    if (!enabled)
        return;
    CudaDeviceRestore restore;
    for (int r = 0; r < count; ++r)
    {
        ggml_backend_t backend = backends[r];
        if (backend == nullptr || !ggml_backend_is_cuda(backend))
            continue;
        auto* ctx = static_cast<ggml_backend_cuda_context*>(backend->context);
        ggml_cuda_set_device(ctx->device);
        (void)cudaStreamQuery(ctx->stream());
    }
#else
    (void)backends;
    (void)count;
#endif
}

int tp_cuda_wait_backends(ggml_backend_t const* backends, int count, int timeout_ms)
{
    if (count > TSG_MAX_DEVICES)
        return -1;
    for (int r = 0; r < count; ++r)
        if (backends[r] == nullptr || !ggml_backend_is_cuda(backends[r]))
            return -1;
    CudaDeviceRestore restore;
    bool done[TSG_MAX_DEVICES] = {};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;)
    {
        bool all = true;
        for (int r = 0; r < count; ++r)
        {
            if (done[r])
                continue;
            auto* ctx = static_cast<ggml_backend_cuda_context*>(backends[r]->context);
            ggml_cuda_set_device(ctx->device);
            const cudaError_t rc = cudaStreamQuery(ctx->stream());
            if (rc == cudaSuccess)
                done[r] = true;
            else if (rc != cudaErrorNotReady)
                return -1;          // a real error: the caller's synchronize reports it
            all &= done[r];
        }
        if (all)
            return 1;
        if (std::chrono::steady_clock::now() >= deadline)
            return 0;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

} // namespace tsg
