// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace tsg_dsv41 {

// Sparse mmap page faults may block on network storage. Keep several requests
// in flight without creating threads for every Engram layer or prefill chunk.
// run() completes before returning; concurrent submitters are serialized.
class engram_io_pool {
    std::vector<std::thread> workers_;
    std::mutex submission_, mutex_;
    std::condition_variable ready_, done_;
    std::function<void(size_t)> job_;
    std::atomic<size_t> cursor_{0};
    size_t count_ = 0, pending_ = 0, generation_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;

    void consume() noexcept {
        try {
            for (;;) {
                const size_t i = cursor_.fetch_add(1, std::memory_order_relaxed);
                if (i >= count_) break;
                job_(i);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
    }

    void worker() {
        size_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            lock.unlock();
            consume();
            lock.lock();
            if (--pending_ == 0) done_.notify_one();
        }
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        ready_.notify_all();
        for (auto & thread : workers_) if (thread.joinable()) thread.join();
    }

public:
    explicit engram_io_pool(unsigned threads) {
        if (threads < 1 || threads > 32) throw std::runtime_error("Engram I/O threads must be in [1, 32]");
        try {
            for (unsigned i = 1; i < threads; ++i) workers_.emplace_back([this] { worker(); });
        } catch (...) { stop(); throw; }
    }
    ~engram_io_pool() { stop(); }
    engram_io_pool(const engram_io_pool &) = delete;
    engram_io_pool & operator=(const engram_io_pool &) = delete;

    unsigned threads() const { return unsigned(workers_.size() + 1); }

    template<typename Function> void run(size_t count, Function function) {
        if (!count) return;
        std::unique_lock<std::mutex> submit(submission_);
        if (workers_.empty() || count == 1) {
            for (size_t i = 0; i < count; ++i) function(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = std::move(function);
            count_ = count;
            cursor_.store(0, std::memory_order_relaxed);
            error_ = nullptr;
            pending_ = workers_.size();
            ++generation_;
        }
        ready_.notify_all();
        consume();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return pending_ == 0; });
        job_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

    // Workers prepare independent sources; only the submitting thread consumes
    // ready items (e.g. issues CUDA uploads). Consumption can overlap remaining
    // reads. All workers are joined before return, including either exception
    // path, so caller-owned mappings/callback captures cannot outlive this call.
    template<typename Prepare, typename Consume>
    void run_pipelined(size_t count, Prepare prepare, Consume consume_ready) {
        if (!count) return;
        std::unique_lock<std::mutex> submit(submission_);
        if (workers_.empty() || count == 1) {
            for (size_t i = 0; i < count; ++i) { prepare(i); consume_ready(i); }
            return;
        }
        std::vector<size_t> ready(count);
        size_t available = 0, consumed = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = [&](size_t i) {
                prepare(i);
                {
                    std::lock_guard<std::mutex> completed(mutex_);
                    ready[available++] = i;
                }
                done_.notify_one();
            };
            count_ = count;
            cursor_.store(0, std::memory_order_relaxed);
            error_ = nullptr;
            pending_ = workers_.size();
            ++generation_;
        }
        ready_.notify_all();
        std::exception_ptr consumer_error;
        std::unique_lock<std::mutex> lock(mutex_);
        while (consumed < count) {
            done_.wait(lock, [&] { return consumed < available || error_ || pending_ == 0; });
            if (error_) break;
            if (consumed == available) break;
            const size_t i = ready[consumed++];
            lock.unlock();
            try { consume_ready(i); }
            catch (...) { consumer_error = std::current_exception(); }
            lock.lock();
            if (consumer_error) break;
        }
        done_.wait(lock, [&] { return pending_ == 0; });
        job_ = nullptr;
        if (consumer_error) std::rethrow_exception(consumer_error);
        if (error_) std::rethrow_exception(error_);
    }

    // Opt-in load-time warming. Each task walks a contiguous 8 MiB range;
    // touching every page faults it in while allowing filesystem readahead.
    // Pages remain evictable; this neither pins nor copies the entire table.
    uint64_t warm(const void * data, size_t bytes) {
        if (!data && bytes) throw std::runtime_error("Invalid Engram warming range");
        constexpr size_t chunk = 8 * 1024 * 1024, page = 4096;
        const auto * source = static_cast<const volatile uint8_t *>(data);
        std::atomic<uint64_t> checksum{0};
        run(bytes / chunk + (bytes % chunk != 0), [&](size_t index) {
            const size_t start = index * chunk, end = start + std::min(chunk, bytes - start);
            uint64_t sum = 0;
            for (size_t offset = start; offset < end; offset += page) sum += source[offset];
            // A tensor need not begin at a page boundary; its last byte can
            // occupy one more page than the regular stride touched.
            if (end > start) sum += source[end - 1];
            checksum.fetch_add(sum, std::memory_order_relaxed);
        });
        return checksum.load(std::memory_order_relaxed);
    }
};

} // namespace tsg_dsv41
