// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "dsv41_workers.h"
#include <atomic>
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

// Copying a nontrivial callable is permitted to allocate or throw. The original
// pool copied it on each background worker outside its exception boundary.
struct copy_probe {
    std::thread::id owner = std::this_thread::get_id();
    std::atomic<int> * calls;
    std::atomic<int> * copies;
    std::array<char, 256> large_capture{};
    copy_probe(std::atomic<int> & calls, std::atomic<int> & copies) : calls(&calls), copies(&copies) {}
    copy_probe(const copy_probe & other)
        : owner(other.owner), calls(other.calls), copies(other.copies), large_capture(other.large_capture) {
        if (std::this_thread::get_id() != owner) throw std::bad_alloc();
        ++*copies;
    }
    void operator()(int) const { ++*calls; }
};
}

int main() {
    try {
        tsg_dsv41_tp::workers pool(2);
        std::atomic<int> calls{0}, copies{0};
        const std::function<void(int)> callable = copy_probe(calls, copies);
        copies = 0;
        pool.run(callable);
        require(calls == 2, "Not every rank completed the borrowed callable");
        require(copies == 0, "Dispatch copied a potentially throwing callable");

        int checks = 2;
        for (int failed_rank : {0, 1}) {
            std::atomic<unsigned> completed{0};
            bool propagated = false;
            try {
                pool.run([&](int rank) {
                    completed.fetch_or(1U << rank);
                    if (rank == failed_rank) throw std::runtime_error("rank error");
                });
            } catch (const std::runtime_error & error) {
                propagated = std::string(error.what()) == "rank error";
            }
            require(propagated, "Rank error did not reach the synchronous caller");
            require(completed == 3, "Caller returned before every rank finished");
            calls = 0;
            pool.run(callable);
            require(calls == 2 && copies == 0, "Pool did not recover after a rank error");
            checks += 3;
        }

        // Temporary std::function and stack captures remain alive for the
        // entire synchronous run, including when one worker finishes first.
        for (int generation = 0; generation < 64; ++generation) {
            int result[2] = {-1, -1};
            pool.run([&](int rank) { result[rank] = generation * 2 + rank; });
            require(result[0] == generation * 2 && result[1] == generation * 2 + 1,
                    "Worker observed a stale or expired job");
            ++checks;
        }
        std::cout << "Passed " << checks << " TP worker lifetime checks\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
