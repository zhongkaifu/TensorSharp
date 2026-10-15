// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Separate process: the production stream threshold is latched on first use.
// Exhausting graph credit must not allocate through the per-context fallback.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

extern "C" {
    const char* TSGgml_GetLastError();
    int TSGgml_IsBackendAvailable(int backend);
    void TSGgml_ReleaseReuseComputeBuffers();
    void TSGgml_ClearHostBufferCache();
    void TSGgml_Shutdown();
    int TSGgml_AttachSharedCacheBudgetEx(void*,
        std::uint64_t (*)(void*, int, int, std::int64_t),
        int (*)(void*, std::uint64_t), void (*)(void*, std::uint64_t), int);
    int TSGgml_DetachSharedCacheBudget(void*);
    int TSGgml_MoEFFNPrefillSwiGLUQuantF32(
        float*, float*, int, int, int, int, int, const std::int32_t*, const float*,
        void*, int, std::int64_t, std::int64_t, std::int64_t,
        void*, int, std::int64_t, std::int64_t, std::int64_t,
        void*, int, std::int64_t, std::int64_t, std::int64_t,
        const float*, const float*, const float*, int, float, float, int);
}

namespace {
    void require(bool condition, const char* message)
    {
        if (!condition) throw std::runtime_error(message);
    }

    void environment(const char* name, const char* value)
    {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    struct Ledger {
        struct Ticket { std::int64_t bytes; bool committed; };
        std::map<std::uint64_t, Ticket> tickets;
        std::uint64_t next = 0, reserve_calls = 0;
        std::int64_t capacity = 0, reserved = 0, committed = 0;
        bool valid = true, attached = false;

        static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes)
        {
            auto& l = *static_cast<Ledger*>(context);
            ++l.reserve_calls;
            if (rank != 0 || kind != 2 || bytes <= 0) { l.valid = false; return 0; }
            if (bytes > l.capacity - l.reserved - l.committed) return 0;
            const auto token = ++l.next;
            l.tickets.emplace(token, Ticket{bytes, false});
            l.reserved += bytes;
            return token;
        }
        static int commit(void* context, std::uint64_t token)
        {
            auto& l = *static_cast<Ledger*>(context);
            auto entry = l.tickets.find(token);
            if (entry == l.tickets.end() || entry->second.committed) { l.valid = false; return 0; }
            entry->second.committed = true;
            l.reserved -= entry->second.bytes;
            l.committed += entry->second.bytes;
            return 1;
        }
        static void release(void* context, std::uint64_t token)
        {
            auto& l = *static_cast<Ledger*>(context);
            auto entry = l.tickets.find(token);
            if (entry == l.tickets.end()) { l.valid = false; return; }
            (entry->second.committed ? l.committed : l.reserved) -= entry->second.bytes;
            l.tickets.erase(entry);
        }
        bool empty() const { return valid && tickets.empty() && reserved == 0 && committed == 0; }
        ~Ledger()
        {
            // Callback storage outlives all native owners even on failed assertions.
            TSGgml_ReleaseReuseComputeBuffers();
            TSGgml_ClearHostBufferCache();
            if (attached) TSGgml_DetachSharedCacheBudget(this);
        }
    };

    constexpr int Tokens = 38, Width = 256, Experts = 4, Used = 2, Q8 = 8;
    constexpr float Canary = -12345.0f;

    std::vector<std::uint8_t> weights(int seed)
    {
        std::vector<std::uint8_t> result(std::size_t(Width) * Width * Experts / 32 * 34);
        for (std::size_t block = 0; block < result.size() / 34; ++block)
        {
            result[block * 34] = 0; result[block * 34 + 1] = 0x20; // F16 scale 1/128
            for (int k = 0; k < 32; ++k)
                result[block * 34 + 2 + k] = static_cast<std::uint8_t>(
                    static_cast<std::int8_t>((int(block % 19) + k * 3 + seed) % 9 - 4));
        }
        return result;
    }
}

int main()
{
    // Force this 38-token run onto the ordinary full-shape GPU stream. No
    // selected-expert cache, pinned host pages, or process-inherited threshold.
    environment("TS_HOST_MOE_DEVICE_MIN_BATCH", "38");
    environment("TS_HOST_MOE_PIN", "0");
    environment("TS_HOST_MOE_EXPERT_CACHE_MB", "0");
    if (TSGgml_IsBackendAvailable(3) == 0)
    {
        std::fprintf(stderr, "SKIP: CUDA backend unavailable: %s\n", TSGgml_GetLastError());
        return 77;
    }
    int result = 0;
    try
    {
        Ledger ledger;
        require(TSGgml_AttachSharedCacheBudgetEx(&ledger, Ledger::reserve, Ledger::commit,
            Ledger::release, 1) == 1, "could not attach graph budget");
        ledger.attached = true;
        auto gate = weights(1), up = weights(2), down = weights(3);
        std::vector<float> input(Tokens * Width), output(Tokens * Width, Canary), routes(Tokens * Used, 0.5f);
        std::vector<std::int32_t> ids(Tokens * Used);
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = (int(i % 31) - 15) / 16.0f;
        for (int t = 0; t < Tokens; ++t)
            for (int k = 0; k < Used; ++k) ids[t * Used + k] = (t + k) % Experts;
        const auto run = [&] {
            return TSGgml_MoEFFNPrefillSwiGLUQuantF32(input.data(), output.data(), Tokens, Width, Width,
                Experts, Used, ids.data(), routes.data(),
                gate.data(), Q8, Width, Width, static_cast<std::int64_t>(gate.size()),
                up.data(), Q8, Width, Width, static_cast<std::int64_t>(up.size()),
                down.data(), Q8, Width, Width, static_cast<std::int64_t>(down.size()),
                nullptr, nullptr, nullptr, 0, 1.702f, 7.0f, /*run_on_cpu=*/1);
        };

        std::vector<float> reference;
        for (int cycle = 0; cycle < 2; ++cycle)
        {
            ledger.capacity = 0;
            const auto before = ledger.reserve_calls;
            std::fill(output.begin(), output.end(), Canary);
            require(run() == 0, "streamed MoE bypassed exhausted graph credit through its fallback");
            require(ledger.reserve_calls - before >= 2,
                "test did not reach BOTH gallocr and per-context graph reservation attempts");
            require(ledger.empty(), "denied stream left a graph reservation or owner");
            require(std::all_of(output.begin(), output.end(), [](float x) { return x == Canary; }),
                "denied stream modified caller output");

            ledger.capacity = 64LL << 20;
            require(run() == 1, "stream did not recover after quota became available");
            require(ledger.valid && ledger.committed > 0 && ledger.reserved == 0,
                "successful streamed graph did not retain its actual buffer charge");
            require(std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x) && x != Canary; })
                && std::any_of(output.begin(), output.end(), [](float x) { return x != 0; }),
                "successful streamed graph returned invalid output");
            if (reference.empty()) reference = output;
            else require(std::memcmp(reference.data(), output.data(), output.size() * sizeof(float)) == 0,
                "recreated stream changed identical-input outputs");
            require(TSGgml_DetachSharedCacheBudget(&ledger) == 0,
                "live streamed allocator released callback lifetime too early");
            TSGgml_ReleaseReuseComputeBuffers();
            require(ledger.empty(), "physical graph cleanup did not return all credit");
        }
        require(TSGgml_DetachSharedCacheBudget(&ledger) == 1, "empty scope could not detach");
        ledger.attached = false;
        std::puts("PASS: forced N38 CUDA stream rejects exhausted quota, preserves output, recovers and refunds");
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "FAIL: %s; native: %s\n", error.what(), TSGgml_GetLastError());
        result = 1;
    }
    TSGgml_Shutdown();
    return result;
}
