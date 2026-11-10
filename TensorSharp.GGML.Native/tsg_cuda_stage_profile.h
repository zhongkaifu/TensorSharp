// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Where the time inside a fused CUDA node goes.
//
// The node profiler (TS_GGML_NODE_PROFILE) times a fused node as one unit, but a
// node such as the Q4_K GEMM is several launches on one stream: the LoRA factors,
// the weight expansion, the activation narrowing and the GEMM itself. With
// TS_H3_STAGE_PROFILE=1 a Clock waits for the default stream at each lap() and
// charges the time since the previous lap to a named stage; dump() prints the
// totals and clears them. The waits serialize launches that otherwise overlap on
// the host, so read the stages against each other, not as the node's unprofiled time.
#pragma once

#if defined(TSG_GGML_USE_CUDA)

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace tsg_stage {

inline bool enabled()
{
    static const bool on = [] {
        const char* e = std::getenv("TS_H3_STAGE_PROFILE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

struct Table
{
    std::mutex mu;
    struct Row { double seconds = 0.0; long long calls = 0; };
    std::map<std::string, Row> rows;
};

inline Table& table()
{
    static Table t;
    return t;
}

class Clock
{
public:
    Clock() : on_(enabled())
    {
        if (!on_) return;
        cudaStreamSynchronize(nullptr);
        last_ = std::chrono::steady_clock::now();
    }

    void lap(const char* stage)
    {
        if (!on_) return;
        cudaStreamSynchronize(nullptr);
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - last_).count();
        last_ = now;
        Table& t = table();
        std::lock_guard<std::mutex> lock(t.mu);
        Table::Row& row = t.rows[stage];
        row.seconds += seconds;
        row.calls++;
    }

private:
    bool on_;
    std::chrono::steady_clock::time_point last_;
};

inline void dump(const char* tag)
{
    if (!enabled()) return;
    Table& t = table();
    std::lock_guard<std::mutex> lock(t.mu);
    for (const auto& [stage, row] : t.rows)
        std::fprintf(stderr, "[h3-stage] tag=%s stage=\"%s\" calls=%lld seconds=%.3f ms_per_call=%.3f\n", tag,
                     stage.c_str(), row.calls, row.seconds, row.calls > 0 ? 1e3 * row.seconds / row.calls : 0.0);
    std::fflush(stderr);
    t.rows.clear();
}

} // namespace tsg_stage

#endif // TSG_GGML_USE_CUDA
