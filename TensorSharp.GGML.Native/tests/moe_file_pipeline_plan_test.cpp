// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_moe_file_pipeline_plan.h"
#include "../ggml_ops_bounded_read_pipeline.h"
#include "../ggml_ops_parallel_read_pipeline.h"
#include "../dsv41_engram_io.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <future>

namespace {
int checks = 0;
void require(bool value) { ++checks; if (!value) throw std::runtime_error("Packing assertion failed"); }
template<class Action> void fails(Action action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    require(threw);
}
void exact_patterns() {
    for (std::size_t slot : {std::size_t(1), std::size_t(7), std::size_t(64), std::size_t(128)})
        for (std::size_t stride : {std::size_t(1), std::size_t(3), std::size_t(31), std::size_t(137)})
            for (unsigned mask = 1; mask < 512; ++mask) {
                tsg::HostMoeFilePipelinePlan plan(slot);
                std::array<std::vector<std::uint8_t>, 3> sources, expected, actual;
                std::array<std::size_t, 3> bytes{};
                for (int part = 0; part < 3; ++part) {
                    const auto width = stride + part;
                    sources[part].resize(9 * width);
                    for (std::size_t i = 0; i < sources[part].size(); ++i)
                        sources[part][i] = static_cast<std::uint8_t>(i * 17 + part * 37);
                    for (unsigned expert = 0; expert < 9; ++expert) if (mask & (1u << expert)) {
                        plan.append(part, expert * width, expected[part].size(), width);
                        expected[part].insert(expected[part].end(), sources[part].begin() + expert * width,
                            sources[part].begin() + (expert + 1) * width);
                    }
                    bytes[part] = expected[part].size(); actual[part].resize(bytes[part]);
                }
                plan.validate(bytes);
                std::array<std::size_t, 3> covered{};
                std::vector<std::uint8_t> storage(slot + 2, 0xa5);
                for (std::size_t i = 0; i < plan.transfers().size(); ++i) {
                    const auto& transfer = plan.transfers()[i];
                    require(transfer.destination == covered[transfer.part]
                        && transfer.bytes > 0 && transfer.bytes <= slot);
                    plan.read_transfer(i, storage.data() + 1, [] {},
                        [&](int part, auto source, auto* target, auto count) {
                            require(part == transfer.part && source <= sources[part].size()
                                && count <= sources[part].size() - source);
                            std::memcpy(target, sources[part].data() + source, count);
                        });
                    require(storage.front() == 0xa5 && storage.back() == 0xa5);
                    std::memcpy(actual[transfer.part].data() + transfer.destination, storage.data() + 1, transfer.bytes);
                    covered[transfer.part] += transfer.bytes;
                }
                require(actual == expected && covered == bytes);
                const auto gates = (bytes[0] + slot - 1) / slot + (bytes[1] + slot - 1) / slot;
                require(plan.gate_transfers() == gates
                    && plan.transfers().size() == gates + (bytes[2] + slot - 1) / slot);
            }
}
void rejects_missing_or_duplicate_bytes() {
    fails([] { tsg::HostMoeFilePipelinePlan plan(0); });
    for (auto part : {-1, 1, 3}) fails([&] { tsg::HostMoeFilePipelinePlan plan(8); plan.append(part, 0, 0, 1); });
    fails([] { tsg::HostMoeFilePipelinePlan plan(8); plan.append(0, 0, 0, 0); });
    fails([] { tsg::HostMoeFilePipelinePlan plan(8); plan.append(0, 0, 1, 2); });
    fails([] { tsg::HostMoeFilePipelinePlan plan(8); plan.append(0, 0, 0, 2); plan.append(0, 3, 1, 2); });
    fails([] { tsg::HostMoeFilePipelinePlan plan(8); plan.append(0, 0, 0, 2); plan.append(0, 3, 3, 2); });
    fails([] { tsg::HostMoeFilePipelinePlan plan(8); plan.append(0, 0, 0, 2); plan.append(2, 0, 0, 2); });
    fails([] { tsg::HostMoeFilePipelinePlan plan(8); plan.append(0, 0, 0, 2); plan.append(1, 0, 0, 2); plan.append(0, 3, 2, 2); });
    fails([] { tsg::HostMoeFilePipelinePlan plan(8); plan.append(0, (std::numeric_limits<std::size_t>::max)(), 0, 2); });
    fails([] { tsg::HostMoeFilePipelinePlan plan(8); plan.append(0, 0, 0, 2); plan.validate({2, 0, 0}); });
    tsg::HostMoeFilePipelinePlan plan(8);
    for (int part = 0; part < 3; ++part) plan.append(part, 0, 0, 2);
    fails([&] { plan.validate({2, 2, 1}); });
    fails([&] { plan.read_transfer(0, nullptr, [] {}, [](auto,auto,auto,auto) {}); });
    std::uint8_t data[8];
    fails([&] { plan.read_transfer(3, data, [] {}, [](auto,auto,auto,auto) {}); });
}
void fragmented_actual_geometry() {
    constexpr std::size_t slot = 8u << 20, expert = 921600;
    tsg::HostMoeFilePipelinePlan plan(slot);
    for (int part = 0; part < 3; ++part)
        for (std::size_t id = 0; id < 128; ++id) plan.append(part, (2 * id) * expert, id * expert, expert);
    plan.validate({128 * expert, 128 * expert, 128 * expert});
    require(plan.transfers().size() == 45 && plan.gate_transfers() == 30);
    require(plan.reads().size() == 426); // Splits keep every source gap, compact DMA fills slots.
}
void cancellation_never_publishes_partial_slot() {
    tsg_dsv41::engram_io_pool pool(4);
    for (int failure : {0, 1}) {
        tsg::HostMoeFilePipelinePlan plan(8);
        for (int part = 0; part < 3; ++part) {
            plan.append(part, 1, 0, 2); plan.append(part, 8, 2, 2);
        }
        plan.validate({4, 4, 4});
        std::array<std::array<std::uint8_t, 8>, 2> slots{};
        std::array<bool, 3> published{};
        std::atomic<int> readers{0};
        bool threw = false;
        try {
            tsg::run_bounded_read_pipeline(pool, plan.transfers().size(), 2, [](auto) {},
                [&](std::size_t index, std::size_t slot) {
                    ++readers;
                    struct Exit { std::atomic<int>& readers; ~Exit() { --readers; } } exit{readers};
                    int ranges = 0;
                    plan.read_transfer(index, slots[slot].data(), [&] {
                        if (failure == 0 && index == 0 && ranges++ == 1) throw std::runtime_error("Mid-slot cancellation");
                    }, [&](int part, auto source, auto* target, auto bytes) {
                        if (failure == 1 && index == 0 && source == 8) throw std::runtime_error("Second source range failed");
                        std::memset(target, part + 1, bytes);
                    });
                }, [&](std::size_t index, std::size_t) { published[index] = true; });
        } catch (const std::runtime_error&) { threw = true; }
        require(threw && readers == 0 && !published[0]);
        int reused = 0;
        tsg::run_bounded_read_pipeline(pool, 7, 2, [](auto) {}, [](auto,auto) {}, [&](auto,auto) { ++reused; });
        require(reused == 7);
    }
}
void parallel_reads_preserve_complete_slots_and_dma_fences() {
    for (unsigned threads : {1u, 2u, 4u, 8u, 16u}) for (std::size_t piece : {1u, 7u, 33u}) {
        tsg_dsv41::engram_io_pool pool(threads);
        tsg::HostMoeFilePipelinePlan plan(64);
        std::array<std::vector<std::uint8_t>, 3> source, expected, actual;
        for (int part = 0; part < 3; ++part) {
            source[part].resize(160);
            for (std::size_t i = 0; i < source[part].size(); ++i) source[part][i] = i * 31 + part * 17;
            for (std::size_t i = 0; i < 8; ++i) {
                plan.append(part, i * 19, i * 13, 13);
                expected[part].insert(expected[part].end(), source[part].begin() + i * 19,
                    source[part].begin() + i * 19 + 13);
            }
            actual[part].resize(expected[part].size());
        }
        plan.validate({104, 104, 104});
        std::vector<std::size_t> counts;
        for (std::size_t i = 0; i < plan.transfers().size(); ++i) counts.push_back(plan.read_job_count(i, piece));
        std::array<std::array<std::uint8_t, 66>, 2> storage;
        for (auto& slot : storage) slot.fill(0xa5);
        std::array<std::future<void>, 2> dma;
        std::array<std::atomic<std::size_t>, 2> owner;
        std::vector<int> preparation(counts.size()), publication(counts.size());
        std::atomic<int> active{0}, maximum{0};
        std::mutex gate_mutex;
        std::condition_variable gate_changed;
        unsigned arrived = 0, target = std::min(4u, threads == 1 ? 1 : threads - 1);
        tsg::run_bounded_parallel_read_pipeline(pool, counts, 2,
            [&](std::size_t index, std::size_t slot) {
                // A deliberately delayed DMA must finish before any next writer
                // can change this same physical ring slot.
                if (dma[slot].valid()) dma[slot].get();
                owner[slot].store(index); ++preparation[index];
            }, [&](std::size_t index, std::size_t job, std::size_t slot) {
                const int now = ++active;
                int old = maximum.load();
                while (old < now && !maximum.compare_exchange_weak(old, now)) {}
                struct Exit { std::atomic<int>& active; ~Exit() { --active; } } exit{active};
                if (index == 0 && job < target) {
                    std::unique_lock<std::mutex> lock(gate_mutex);
                    ++arrived; gate_changed.notify_all();
                    gate_changed.wait(lock, [&] { return arrived == target; });
                }
                if (owner[slot].load() != index) throw std::runtime_error("Ring owner changed during a read");
                plan.read_job(index, job, piece, storage[slot].data() + 1, [] {},
                    [&](int part, auto offset, auto* destination, auto bytes) {
                        if (bytes > piece || offset > source[part].size() || bytes > source[part].size() - offset)
                            throw std::runtime_error("Parallel source read lost its bounds");
                        std::memcpy(destination, source[part].data() + offset, bytes);
                    });
            }, [&](std::size_t index, std::size_t slot) {
                ++publication[index];
                const auto transfer = plan.transfers()[index];
                dma[slot] = std::async(std::launch::async, [&, transfer, slot] {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                    std::memcpy(actual[transfer.part].data() + transfer.destination,
                        storage[slot].data() + 1, transfer.bytes);
                });
            });
        for (auto& fence : dma) if (fence.valid()) fence.get();
        require(actual == expected && active == 0 && maximum >= static_cast<int>(target));
        for (std::size_t i = 0; i < counts.size(); ++i) require(preparation[i] == 1 && publication[i] == 1);
        for (const auto& slot : storage) require(slot.front() == 0xa5 && slot.back() == 0xa5);
        require(plan.read_job_count(0, piece) == counts[0]);
        fails([&] { plan.read_job_count(0, 0); });
        fails([&] { plan.read_job_count(counts.size(), piece); });
        fails([&] { plan.read_job(0, counts[0], piece, storage[0].data(), [] {}, [](auto,auto,auto,auto) {}); });
    }
}
void parallel_read_failures_join_without_partial_publication() {
    tsg_dsv41::engram_io_pool pool(8);
    for (int failure = 0; failure < 6; ++failure) {
        const std::vector<std::size_t> counts{8, 8, 8, 8, 8, 8};
        std::array<bool, 6> published{};
        std::atomic<int> readers{0};
        bool threw = false;
        try {
            tsg::run_bounded_parallel_read_pipeline(pool, counts, 2,
                [&](std::size_t index, std::size_t) {
                    if (failure == 0 && index == 0) throw std::runtime_error("Transfer fence failed");
                }, [&](std::size_t index, std::size_t job, std::size_t) {
                    ++readers;
                    struct Exit { std::atomic<int>& readers; ~Exit() { --readers; } } exit{readers};
                    if (index == 0 && job == (failure == 1 ? 0u : 3u)) {
                        if (failure == 1 || failure == 2) throw std::runtime_error("Read/cancellation failed");
                        if (failure == 3) throw 17;
                    }
                }, [&](std::size_t index, std::size_t) {
                    if (index == 0 && failure == 4) throw std::runtime_error("Consumer submission failed");
                    if (index == 0 && failure == 5) throw 23;
                    published[index] = true;
                });
        } catch (...) { threw = true; }
        require(threw && readers == 0 && !published[0]);
        int reused = 0;
        tsg::run_bounded_parallel_read_pipeline(pool, counts, 2, [](auto,auto) {}, [](auto,auto,auto) {},
            [&](auto,auto) { ++reused; });
        require(reused == 6);
    }
    fails([&] { tsg::run_bounded_parallel_read_pipeline(pool, {}, 0, [](auto,auto) {}, [](auto,auto,auto) {}, [](auto,auto) {}); });
    fails([&] { tsg::run_bounded_parallel_read_pipeline(pool, {0}, 2, [](auto,auto) {}, [](auto,auto,auto) {}, [](auto,auto) {}); });
    fails([&] { tsg::run_bounded_parallel_read_pipeline(pool, {(std::numeric_limits<std::size_t>::max)(), 1}, 2,
        [](auto,auto) {}, [](auto,auto,auto) {}, [](auto,auto) {}); });
}
}
int main() {
    try {
        exact_patterns(); rejects_missing_or_duplicate_bytes(); fragmented_actual_geometry();
        cancellation_never_publishes_partial_slot();
        parallel_reads_preserve_complete_slots_and_dma_fences();
        parallel_read_failures_join_without_partial_publication();
        std::printf("PASS file-pipeline packing: %d checks, exact fragmented coverage, tails, projection sources, bounds and joined mid-slot failures\n", checks);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
