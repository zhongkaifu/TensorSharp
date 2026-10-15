// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_bounded_read_pipeline.h"
#include "dsv41_engram_io.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <future>
#include <thread>

static int checks;
static void require(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
int main() {
    try {
        tsg_dsv41::engram_io_pool pool(8);
        for (int failure : {-1, 0, 1, 2}) {
            std::array<std::size_t, 2> payload{};
            std::array<std::future<void>, 2> transfers;
            std::array<std::atomic<bool>, 2> in_flight{};
            std::atomic<int> readers{0}, read_count{0};
            int consumed = 0;
            bool threw = false;
            try {
                tsg::run_bounded_read_pipeline(pool, 41, 2,
                    [&](std::size_t slot) {
                        if (transfers[slot].valid()) transfers[slot].get();
                        if (failure == 0 && read_count >= 3) throw std::runtime_error("fence failure");
                    }, [&](std::size_t index, std::size_t slot) {
                        struct Exit { std::atomic<int>& n; ~Exit() { --n; } } exit{readers};
                        ++readers;
                        if (in_flight[slot]) throw std::runtime_error("slot overwritten before transfer completed");
                        if (failure == 1 && index == 7) throw std::runtime_error("read failure");
                        payload[slot] = index;
                        ++read_count;
                    }, [&](std::size_t index, std::size_t slot) {
                        if (payload[slot] != index) throw std::runtime_error("slot identity changed");
                        if (failure == 2 && index == 6) throw std::runtime_error("transfer submission failure");
                        in_flight[slot] = true;
                        transfers[slot] = std::async(std::launch::async, [&, index, slot] {
                            for (int i = 0; i < 20; ++i) std::this_thread::yield();
                            if (payload[slot] != index) throw std::runtime_error("DMA source changed");
                            in_flight[slot] = false;
                        });
                        ++consumed;
                    });
            } catch (const std::runtime_error&) { threw = true; }
            for (auto& transfer : transfers) if (transfer.valid()) transfer.get();
            require(readers == 0, "return raced an active read worker");
            require(!in_flight[0] && !in_flight[1], "test drain left a transfer alive");
            require(threw == (failure >= 0), "failure did not propagate");
            if (failure < 0) require(consumed == 41 && read_count == 41, "normal tasks skipped/repeated");
            // The same pool remains usable after every error; no detached worker.
            int retried = 0;
            tsg::run_bounded_read_pipeline(pool, 5, 2, [](auto) {}, [](auto,auto) {},
                [&](auto,auto) { ++retried; });
            require(retried == 5, "retry failed after cancellation");
        }
        bool invalid = false;
        try { tsg::run_bounded_read_pipeline(pool, 0, 0, [](auto){}, [](auto,auto){}, [](auto,auto){}); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "zero slots accepted");
        tsg_dsv41::engram_io_pool serial(1);
        int serial_count = 0;
        tsg::run_bounded_read_pipeline(serial, 7, 2, [](auto){}, [](auto,auto){},
            [&](auto,auto){++serial_count;});
        require(serial_count == 7, "serial pool path failed");
        std::printf("PASS bounded read pipeline: %d checks, slot fences, joined failures, retry and serial execution\n", checks);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
