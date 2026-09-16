// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tsg_dsv41_tp {
class workers
{
public:
    explicit workers(int count)
    {
        try
        {
            for (int rank = 1; rank < count; ++rank)
                threads.emplace_back([this, rank] {
                    uint64_t seen = 0;
                    for (;;)
                    {
                        const std::function<void(int)> * current = nullptr;
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            wake.wait(lock, [&] { return stopping || generation != seen; });
                            if (stopping) return;
                            seen = generation;
                            current = job;
                        }
                        invoke(*current, rank);
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            if (--pending == 0) done.notify_one();
                        }
                    }
                });
        }
        catch (...)
        {
            // std::thread may throw after earlier workers started. A partially
            // constructed vector of joinable threads would otherwise terminate
            // the process while the loader tries to report the resource error.
            stop();
            throw;
        }
    }
    ~workers() { stop(); }
    void stop() noexcept
    {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_all();
        for (auto & thread : threads) if (thread.joinable()) thread.join();
    }
    // The executor serializes run/stop/destruction. The callable must permit
    // concurrent invocation; mutable state must be rank-private or synchronized.
    // The caller owns operation until every
    // rank has finished; borrowing it avoids callable copies (and their possible
    // allocation failures) on background threads outside the error boundary.
    void run(const std::function<void(int)> & operation)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            failure = nullptr;
            job = &operation;
            pending = (int) threads.size();
            ++generation;
        }
        wake.notify_all();
        invoke(operation, 0);
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [&] { return pending == 0; });
        job = nullptr;
        if (failure) std::rethrow_exception(failure);
    }
private:
    void invoke(const std::function<void(int)> & operation, int rank)
    {
        try { operation(rank); }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!failure) failure = std::current_exception();
        }
    }
    std::mutex mutex;
    std::condition_variable wake, done;
    std::vector<std::thread> threads;
    const std::function<void(int)> * job = nullptr;
    std::exception_ptr failure;
    uint64_t generation = 0;
    int pending = 0;
    bool stopping = false;
};

} // namespace tsg_dsv41_tp
