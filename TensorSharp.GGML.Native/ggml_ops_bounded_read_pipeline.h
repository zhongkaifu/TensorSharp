// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace tsg {
// Reuses an existing joined I/O pool. A slot becomes reusable after consume has
// submitted its transfer; before_read must wait that slot's transfer fence
// before overwriting its bytes. The slot sequence prevents later tasks from
// starving an earlier projection. No callback survives pool.run_pipelined().
// Either callback failure wakes every waiting producer before the pool joins.
template<class Pool, class BeforeRead, class Read, class Consume>
void run_bounded_read_pipeline(Pool& pool, std::size_t count, std::size_t slots,
    BeforeRead before_read, Read read, Consume consume) {
    if (!slots) throw std::invalid_argument("A read pipeline needs at least one slot");
    if (!count) return;
    std::vector<std::size_t> next(slots);
    for (std::size_t slot = 0; slot < slots; ++slot) next[slot] = slot;
    std::mutex mutex;
    std::condition_variable changed;
    bool stopped = false;
    auto stop = [&] {
        { std::lock_guard<std::mutex> lock(mutex); stopped = true; }
        changed.notify_all();
    };
    pool.run_pipelined(count, [&](std::size_t index) {
        try {
            const auto slot = index % slots;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return stopped || next[slot] == index; });
                if (stopped) return;
            }
            before_read(slot);
            read(index, slot);
        } catch (...) { stop(); throw; }
    }, [&](std::size_t index) {
        try {
            { std::lock_guard<std::mutex> lock(mutex); if (stopped) return; }
            const auto slot = index % slots;
            consume(index, slot);
            {
                std::lock_guard<std::mutex> lock(mutex);
                // Saturate after the final use; do not overflow even for an
                // abstract SIZE_MAX task count in a portable caller.
                next[slot] = count - index <= slots ? count : index + slots;
            }
            changed.notify_all();
        } catch (...) { stop(); throw; }
    });
}
}
