// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_cache_budget.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool value, const char* message)
{
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main()
{
    std::mutex mutex;
    std::int64_t reserved = 0, committed = 0, limit = 1024;
    std::atomic<int> attempted{0}, admitted{0};
    std::atomic<bool> release{false};
    std::vector<std::thread> workers;
    for (int i = 0; i < 32; ++i) workers.emplace_back([&] {
        tsg::CacheAllocationReservation token(mutex, reserved, committed, limit, 64);
        if (token) admitted.fetch_add(1);
        attempted.fetch_add(1);
        while (!release.load()) std::this_thread::yield();
        if (token) {
            std::lock_guard<std::mutex> lock(mutex);
            require(token.fits_locked(64), "another miss stole reserved bytes");
            token.publish_locked(64);
            require(reserved + committed <= limit, "concurrent misses exceeded the cache quota");
        }
    });
    while (attempted.load() != 32) std::this_thread::yield();
    {
        std::lock_guard<std::mutex> lock(mutex);
        require(admitted == 16 && reserved == 1024 && committed == 0,
            "pending allocations were not included in admission");
    }
    release = true;
    for (auto& worker : workers) worker.join();
    require(reserved == 0 && committed == 1024, "publication lost allocation charges");
    committed = 0; // Simulate physical cache release.

    try {
        tsg::CacheAllocationReservation token(mutex, reserved, committed, limit, 768);
        require(bool(token), "rollback setup could not reserve");
        throw std::runtime_error("injected backend allocation failure");
    } catch (const std::runtime_error&) { }
    require(reserved == 0 && committed == 0, "failed allocation retained pending credit");
    {
        tsg::CacheAllocationReservation token(mutex, reserved, committed, limit, 768);
        std::lock_guard<std::mutex> lock(mutex);
        require(!token.fits_locked(2048), "oversized physical buffer was accepted");
    }
    require(reserved == 0 && committed == 0, "rejected allocation did not roll back");
    {
        tsg::CacheAllocationReservation token(mutex, reserved, committed, limit, 768);
        std::lock_guard<std::mutex> lock(mutex);
        require(token.fits_locked(512), "smaller physical buffer could not commit");
        token.publish_locked(512);
    }
    require(reserved == 0 && committed == 512, "physical committed size is incorrect");
    {
        tsg::CacheAllocationReservation token(mutex, reserved, committed, limit, 512);
        std::lock_guard<std::mutex> lock(mutex);
        limit = 600;
        require(!token.fits_locked(512), "capacity reduction was ignored before publication");
    }
    require(reserved == 0 && committed == 512, "reduced-capacity rollback released live bytes");
    committed = 0;
    limit = 0; // Legacy uncapped behavior, with signed-counter overflow protection.
    {
        tsg::CacheAllocationReservation token(mutex, reserved, committed, limit,
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        require(bool(token), "uncapped reservation was rejected");
        tsg::CacheAllocationReservation overflow(mutex, reserved, committed, limit, 1);
        require(!overflow, "reservation counter overflow was accepted");
    }
    require(reserved == 0 && committed == 0, "uncapped rollback leaked accounting");
    std::puts("PASS concurrent cache admission, rollback, physical-size commit, capacity reduction and overflow");
}
