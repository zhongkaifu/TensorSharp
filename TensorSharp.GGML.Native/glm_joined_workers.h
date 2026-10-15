// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <atomic>
#include <cstdio>
#include <exception>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace tsg_glm {

struct file_closer
{
    void operator()(FILE * file) const noexcept { if (file) std::fclose(file); }
};
using unique_file = std::unique_ptr<FILE, file_closer>;

struct start_worker_thread
{
    template<class Work> std::thread operator()(Work && work) const
    {
        return std::thread(std::forward<Work>(work));
    }
};

// A worker allocation/read failure must reach the loader's calling thread,
// where it can become a load refusal. An exception escaping std::thread, or
// unwinding a partially created vector of joinable threads, terminates instead.
// Workers observe stop between jobs; none outlive the caller or its buffers.
template<class Work, class Start = start_worker_thread>
void run_joined_workers(size_t count, std::atomic<bool> & stop, Work && work, Start start = {})
{
    std::vector<std::thread> threads;
    threads.reserve(count); // No running threads yet if allocation fails.
    std::exception_ptr first_error;
    std::atomic_flag error_claimed = ATOMIC_FLAG_INIT;
    const auto capture_error = [&]() noexcept
    {
        // Only the winner writes first_error. Joining every worker before
        // reading it supplies synchronization without allocating an error string.
        if (!error_claimed.test_and_set(std::memory_order_relaxed))
            first_error = std::current_exception();
        stop.store(true, std::memory_order_relaxed);
    };
    const auto guarded = [&]() noexcept
    {
        try { work(); }
        catch (...) { capture_error(); }
    };
    try
    {
        for (size_t i = 0; i < count && !stop.load(std::memory_order_relaxed); ++i)
            threads.emplace_back(start(guarded));
    }
    catch (...) { capture_error(); }
    // All entries own threads created here, and this is their only join site.
    for (auto & thread : threads) thread.join();
    if (first_error) std::rethrow_exception(first_error);
}

} // namespace tsg_glm
