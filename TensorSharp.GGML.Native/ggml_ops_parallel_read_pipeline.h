// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace tsg {
// Flatten independent reads within each admitted slot into the existing pool.
// No nested pool submission is allowed: it would deadlock the pool's single
// submission owner. A transfer is published only after every read succeeds.
// Future transfers wait for the same slot's previous DMA before any overwrite;
// all callbacks are joined before either exception leaves this function.
template<class Pool, class BeforeTransfer, class Read, class Consume>
void run_bounded_parallel_read_pipeline(Pool& pool, const std::vector<std::size_t>& read_counts,
    std::size_t slots, BeforeTransfer before_transfer, Read read, Consume consume) {
    if (!slots) throw std::invalid_argument("A parallel read pipeline needs a slot");
    if (read_counts.empty()) return;
    std::vector<std::size_t> ends;
    ends.reserve(read_counts.size());
    std::size_t total = 0;
    for (const auto count : read_counts) {
        if (!count || count > (std::numeric_limits<std::size_t>::max)() - total)
            throw std::invalid_argument("Invalid parallel read count");
        total += count; ends.push_back(total);
    }
    struct Slot { std::size_t next; bool preparing = false, ready = false; };
    std::vector<Slot> state;
    for (std::size_t slot = 0; slot < slots; ++slot) state.push_back({slot});
    auto remaining = read_counts; // Only the submitting consumer changes it.
    std::mutex mutex;
    std::condition_variable changed;
    bool stopped = false;
    auto stop = [&] {
        { std::lock_guard<std::mutex> lock(mutex); stopped = true; }
        changed.notify_all();
    };
    auto transfer_for = [&](std::size_t job) {
        return static_cast<std::size_t>(std::upper_bound(ends.begin(), ends.end(), job) - ends.begin());
    };
    pool.run_pipelined(total, [&](std::size_t job) {
        try {
            const auto index = transfer_for(job), slot = index % slots;
            bool first = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return stopped || state[slot].next == index; });
                if (stopped) return;
                if (!state[slot].preparing) {
                    state[slot].preparing = true; first = true;
                } else {
                    changed.wait(lock, [&] { return stopped || state[slot].ready; });
                    if (stopped) return;
                }
            }
            if (first) {
                before_transfer(index, slot);
                { std::lock_guard<std::mutex> lock(mutex); state[slot].ready = true; }
                changed.notify_all();
            }
            read(index, job - (index ? ends[index - 1] : 0), slot);
        } catch (...) { stop(); throw; }
    }, [&](std::size_t job) {
        try {
            { std::lock_guard<std::mutex> lock(mutex); if (stopped) return; }
            const auto index = transfer_for(job), slot = index % slots;
            if (--remaining[index]) return;
            consume(index, slot);
            {
                std::lock_guard<std::mutex> lock(mutex);
                state[slot].next = read_counts.size() - index <= slots ? read_counts.size() : index + slots;
                state[slot].preparing = state[slot].ready = false;
            }
            changed.notify_all();
        } catch (...) { stop(); throw; }
    });
}
} // namespace tsg
