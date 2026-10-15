// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "../dsv41_engram.h"
#include "../dsv41_engram_io.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require(bool value, const char * message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
template<typename Function> static void expect_error(Function function) {
    bool caught = false;
    try { function(); } catch (const std::runtime_error &) { caught = true; }
    require(caught, "Expected an Engram I/O error");
}

static void test_joint_lookup() {
    tsg_dsv41::engram_data layout;
    layout.max_ngram_size = 4; layout.n_heads = 8; layout.head_dim = 256;
    for (int layer : {1, 14, 19}) layout.layers.push_back({layer, 1024, {}, {}, {}});
    std::vector<int16_t> quantized(1024 * layout.head_dim);
    std::vector<float> plain(quantized.size());
    for (size_t i = 0; i < quantized.size(); ++i) {
        quantized[i] = int16_t(int(i * 71 % 511) - 255);
        plain[i] = float(int(i * 41 % 509) - 254) / 97;
    }
    auto dequantize = [](const void * source, float * output, size_t count) {
        const auto * values = static_cast<const int16_t *>(source);
        for (size_t i = 0; i < count; ++i) output[i] = float(values[i]) / 127;
    };
    auto copy = [](const void * source, float * output, size_t count) { std::memcpy(output, source, count * sizeof(float)); };
    for (size_t tokens : {size_t(1), size_t(3), size_t(257)}) for (bool mask_images : {false, true}) {
        const size_t rows = tokens * layout.hash_columns(), elements = rows * layout.head_dim;
        std::vector<std::vector<int32_t>> hashes(3, std::vector<int32_t>(rows));
        std::vector<std::vector<float>> expected(3, std::vector<float>(elements));
        std::vector<bool> images(tokens);
        for (size_t i = 0; i < tokens; ++i) images[i] = mask_images && (tokens == 1 || i % 3 == 1);
        for (size_t e = 0; e < 3; ++e) {
            for (size_t i = 0; i < rows; ++i) hashes[e][i] = int32_t((i * 257 + i / 3 + e * 117) % 1024);
            if (e == 1) layout.lookup(e, plain.data(), 1024, layout.head_dim * sizeof(float), hashes[e].data(), tokens, expected[e].data(), copy);
            else layout.lookup(e, quantized.data(), 1024, layout.head_dim * sizeof(int16_t), hashes[e].data(), tokens, expected[e].data(), dequantize);
            for (size_t i = 0; i < rows; ++i) if (images[i / layout.hash_columns()])
                std::fill_n(expected[e].data() + i * layout.head_dim, layout.head_dim, 0.0f);
        }
        for (unsigned threads : {1, 16}) for (size_t group_size : {size_t(1), size_t(2), size_t(3)}) {
            tsg_dsv41::engram_io_pool pool(threads);
            std::vector<std::vector<float>> staging(group_size, std::vector<float>(elements));
            std::vector<std::vector<float>> actual(3);
            std::vector<std::function<void(size_t)>> prepared;
            std::atomic<size_t> reads{0};
            // Build callbacks in a helper whose parameter stack expires before
            // any worker executes. Both the dequantizer and scalar metadata
            // must be owned by the returned prepared callable.
            auto prepare = [&](size_t e, const void * table, size_t stride, bool is_plain) {
                auto decode = [is_plain, dequantize, copy, &reads](const void * source, float * output, size_t count) {
                    ++reads;
                    if (is_plain) copy(source, output, count); else dequantize(source, output, count);
                };
                return layout.prepare_lookup(e, table, 1024, stride, hashes[e].data(), tokens,
                    staging[e % group_size].data(), decode);
            };
            auto prepare_all = [&] {
                prepared.clear();
                for (size_t e = 0; e < 3; ++e) prepared.emplace_back(prepare(e,
                    e == 1 ? static_cast<const void *>(plain.data()) : quantized.data(),
                    layout.head_dim * (e == 1 ? sizeof(float) : sizeof(int16_t)), e == 1));
            };
            const int32_t valid_last = hashes[2].back();
            hashes[2].back() = 1024;
            expect_error(prepare_all);
            require(reads == 0, "Invalid later table started earlier row reads");
            hashes[2].back() = valid_last;
            prepare_all();
            auto run = [&](bool fail) {
                for (size_t first = 0; first < prepared.size(); first += group_size) {
                    const size_t count = std::min(group_size, prepared.size() - first);
                    pool.run(count * rows, [&](size_t task) {
                        const size_t e = first + task % count, row = task / count;
                        if (fail && e == 2 && row == 3) throw std::runtime_error("joint later table read");
                        if (images[row / layout.hash_columns()])
                            std::fill_n(staging[e % group_size].data() + row * layout.head_dim, layout.head_dim, 0.0f);
                        else prepared[e](row);
                    });
                    // Model uploads happen here, after this group's workers
                    // finish and before the bounded staging buffers are reused.
                    for (size_t e = first; e < first + count; ++e) actual[e] = staging[e % group_size];
                }
            };
            expect_error([&] { run(true); });
            reads = 0;
            run(false);
            require(reads == size_t(std::count(images.begin(), images.end(), false)) * layout.hash_columns() * 3,
                    "Joint lookup read an image row or skipped a text row");
            for (size_t e = 0; e < 3; ++e)
                require(std::memcmp(expected[e].data(), actual[e].data(), elements * sizeof(float)) == 0,
                        "Joint/grouped mixed-type lookup changed output bytes");
        }
    }
}

static void test_pipeline() {
    using namespace std::chrono;
    for (unsigned threads : {1, 2, 8, 16}) {
        tsg_dsv41::engram_io_pool pool(threads);
        const auto caller = std::this_thread::get_id();
        std::mutex gate;
        std::condition_variable opened;
        bool consumed_first = false, overlapped = false;
        std::vector<int> prepared(24, 0), consumed(24, 0);
        std::atomic<int> finished{0};
        pool.run_pipelined(prepared.size(), [&](size_t i) {
            if (i != 0) {
                std::unique_lock<std::mutex> lock(gate);
                if (!opened.wait_for(lock, seconds(2), [&] { return consumed_first; }))
                    throw std::runtime_error("Pipeline waited for all reads before consuming the first ready item");
            }
            prepared[i] = 17 + int(i);
            ++finished;
        }, [&](size_t i) {
            require(std::this_thread::get_id() == caller, "Pipeline consumer left its submitting thread");
            require(prepared[i] == 17 + int(i), "Pipeline consumed an incomplete source");
            ++consumed[i];
            if (i == 0) {
                overlapped = finished < int(prepared.size());
                {
                    std::lock_guard<std::mutex> lock(gate);
                    consumed_first = true;
                }
                opened.notify_all();
            }
        });
        require(overlapped, "Pipeline had no read/consume overlap opportunity");
        for (int calls : consumed) require(calls == 1, "Pipeline consumed a source more or less than once");
        for (bool fail_prepare : {true, false}) {
            std::atomic<int> active{0};
            expect_error([&] {
                pool.run_pipelined(24, [&](size_t i) {
                    struct Guard {
                        std::atomic<int>& active;
                        Guard(std::atomic<int>& a) : active(a) { ++active; }
                        ~Guard() { --active; }
                    } guard(active);
                    if (fail_prepare && i == 3) throw std::runtime_error("prepare");
                }, [&](size_t) {
                    if (!fail_prepare) throw std::runtime_error("consume");
                });
            });
            require(active == 0, "Pipeline returned before failed-job workers released their source owners");
            std::fill(consumed.begin(), consumed.end(), 0);
            pool.run_pipelined(24, [](size_t) {}, [&](size_t i) { ++consumed[i]; });
            for (int calls : consumed) require(calls == 1, "Pipeline did not recover after an exception");
        }
        std::atomic<int> concurrent_calls{0};
        std::vector<std::thread> submitters;
        for (int j = 0; j < 4; ++j) submitters.emplace_back([&, j] {
            if (j % 2 == 0) pool.run_pipelined(24, [](size_t) {}, [&](size_t) { ++concurrent_calls; });
            else pool.run(24, [&](size_t) { ++concurrent_calls; });
        });
        for (auto& thread : submitters) thread.join();
        require(concurrent_calls == 96, "Mixed pipeline/plain pool submissions lost work");
        pool.run_pipelined(0, [](size_t) { throw std::runtime_error("empty prepare"); },
            [](size_t) { throw std::runtime_error("empty consume"); });
    }
}



int main() {
    using tsg_dsv41::engram_io_pool;
    expect_error([] { engram_io_pool invalid(0); });
    expect_error([] { engram_io_pool invalid(33); });
    test_joint_lookup();
    test_pipeline();
    tsg_dsv41::engram_data layout;
    // Published V4.1 geometry: 24 independently hashed 256-element rows
    // per token. Cover decode and the former serial small-prefill path.
    layout.max_ngram_size = 4; layout.n_heads = 8; layout.head_dim = 256;
    layout.layers.push_back({1, 1024, {}, {}, {}});
    std::vector<int16_t> table(1024 * layout.head_dim);
    for (size_t i = 0; i < table.size(); ++i) table[i] = int16_t(int((i * 71) % 511) - 255);
    std::vector<int32_t> hashes(257 * layout.hash_columns());
    for (size_t i = 0; i < hashes.size(); ++i) hashes[i] = int32_t((i * 257 + i / 3) % 1024);
    auto dequantize = [](const void * source, float * output, size_t count) {
        const auto * values = static_cast<const int16_t *>(source);
        for (size_t i = 0; i < count; ++i) output[i] = float(values[i]) / 127;
    };
    const size_t row_bytes = layout.head_dim * sizeof(int16_t);
    std::vector<float> expected(hashes.size() * layout.head_dim), actual(expected.size());
    layout.lookup(0, table.data(), 1024, row_bytes, hashes.data(), 257, expected.data(), dequantize);
    for (unsigned threads : {1, 2, 8, 16, 32}) {
        engram_io_pool pool(threads);
        for (size_t tokens : {size_t(1), size_t(3), size_t(257)}) {
            const size_t elements = tokens * layout.hash_columns() * layout.head_dim;
            for (int repeat = 0; repeat < 10; ++repeat) {
                std::fill(actual.begin(), actual.end(), -98765.0f);
                layout.lookup(0, table.data(), 1024, row_bytes, hashes.data(), tokens, actual.data(), dequantize,
                    [&](size_t count, auto row) { pool.run(count, row); });
                require(std::memcmp(actual.data(), expected.data(), elements * sizeof(float)) == 0,
                        "Parallel Engram lookup changed row order or arithmetic");
                require(std::all_of(actual.begin() + elements, actual.end(), [](float value) { return value == -98765.0f; }),
                        "Small Engram lookup wrote beyond its output rows");
            }
            // Validate every hash before submitting any row, including a
            // missing last row after otherwise valid decode/prefill hashes.
            auto invalid = hashes;
            const size_t last = tokens * layout.hash_columns() - 1;
            for (int32_t missing : {-1, 1024}) {
                invalid[last] = missing;
                std::atomic<int> calls{0};
                expect_error([&] {
                    layout.lookup(0, table.data(), 1024, row_bytes, invalid.data(), tokens, actual.data(),
                        [&](const void *, float *, size_t) { ++calls; },
                        [&](size_t count, auto row) { pool.run(count, row); });
                });
                require(calls == 0, "Invalid Engram hash submitted partial work");
            }
        }
        for (size_t count : {size_t(24), size_t(72), size_t(129)}) {
            expect_error([&] { pool.run(count, [](size_t i) { if (i == 13) throw std::runtime_error("test"); }); });
            std::vector<std::atomic<int>> recovered(count);
            for (auto & visits : recovered) visits = 0;
            pool.run(count, [&](size_t i) { ++recovered[i]; });
            for (const auto & visits : recovered) require(visits == 1, "Small pool job did not recover after exception");
        }
        expect_error([&] {
            layout.lookup(0, nullptr, 1024, row_bytes, hashes.data(), 1, actual.data(), dequantize,
                [&](size_t count, auto row) { pool.run(count, row); });
        });
        std::vector<std::atomic<int>> visits(257);
        for (auto & count : visits) count = 0;
        pool.run(visits.size(), [&](size_t i) { ++visits[i]; });
        for (const auto & count : visits) require(count == 1, "Pool did not recover after worker exception");
        std::vector<std::thread> submitters;
        for (int i = 0; i < 4; ++i) submitters.emplace_back([&] {
            for (int repeat = 0; repeat < 5; ++repeat) pool.run(visits.size(), [&](size_t row) { ++visits[row]; });
        });
        for (auto & thread : submitters) thread.join();
        for (const auto & count : visits) require(count == 21, "Concurrent pool submissions overlapped incorrectly");
    }
    std::vector<uint8_t> pages(17 * 1024 * 1024 + 13);
    for (size_t i = 0; i < pages.size(); ++i) pages[i] = uint8_t(i * 19 + i / 17);
    const auto original = pages;
    engram_io_pool serial(1), parallel(16);
    const auto checksum = serial.warm(pages.data() + 13, pages.size() - 13);
    require(checksum == parallel.warm(pages.data() + 13, pages.size() - 13), "Warming checksum changed");
    require(pages == original, "Warming modified table bytes");
    require(parallel.warm(nullptr, 0) == 0, "Empty warming failed");
    expect_error([&] { parallel.warm(nullptr, 1); });
    // Model delayed page reads deterministically; timing is diagnostic, while
    // peak overlap verifies the optimization without a flaky speed threshold.
    for (size_t rows : {size_t(24), size_t(72)}) for (auto * pool : {&serial, &parallel}) {
        std::atomic<int> active{0}, peak{0};
        const auto start = std::chrono::steady_clock::now();
        pool->run(rows, [&](size_t) {
            const int current = ++active;
            int previous = peak.load();
            while (previous < current && !peak.compare_exchange_weak(previous, current)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            --active;
        });
        require(peak <= int(pool->threads()), "Pool exceeded its worker bound");
        if (pool->threads() > 1) require(peak > 1, "Delayed I/O was still serialized");
        std::printf("Simulated blocked reads: rows=%zu threads=%u peak=%d elapsed_ms=%.2f\n", rows, pool->threads(), int(peak),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::puts("Engram 1/3/257-token serial/parallel/warm, invalid rows, lifecycle, concurrent submission and exception tests passed");
}
