// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <array>
#include <cstdint>

namespace tsg_glm {
constexpr int kBatchedExecutionRows = 16;

// One successful logical scheduler graph, potentially split into several
// backend dispatches. Records are model-owned and read under the executor lock.
struct batched_execution {
    std::uint64_t id = 0;
    int rows = 0, nodes = 0, backend_splits = 0;
    bool graph_reused = false;
    std::array<std::int32_t, kBatchedExecutionRows> slots{}, before{}, after{};
};

class batched_execution_recorder {
    std::uint64_t accepted_ = 0;
    batched_execution last_{};
    bool dispatch_started_ = false;
    int attempted_rows_ = 0;
    std::array<std::int32_t, kBatchedExecutionRows> attempted_slots_{};
public:
    void begin_attempt() noexcept { last_ = {}; dispatch_started_ = false; attempted_rows_ = 0; }
    bool begin_dispatch(int rows, const std::int32_t* slots) noexcept {
        if (rows < 2 || rows > kBatchedExecutionRows || !slots) return false;
        dispatch_started_ = true;
        attempted_rows_ = rows;
        for (int i = 0; i < rows; ++i) attempted_slots_[i] = slots[i];
        return true;
    }
    // A failed compute can have written any participating slot even when no
    // logits/head was published. Serial retry must require a reset of all rows.
    bool failure_needs_reset() const noexcept { return dispatch_started_ && last_.id == 0; }
    int attempted_rows() const noexcept { return attempted_rows_; }
    std::int32_t attempted_slot(int i) const noexcept { return attempted_slots_[i]; }
    const batched_execution& last() const noexcept { return last_; }
    bool record_success(int rows, int nodes, int splits, bool reused,
        const std::int32_t* slots, const std::int32_t* before, const std::int32_t* after) noexcept {
        last_ = {};
        if (rows < 2 || rows > kBatchedExecutionRows || nodes <= 0 || splits <= 0
            || !slots || !before || !after) return false;
        for (int i = 0; i < rows; ++i) {
            if (slots[i] < 0 || before[i] < 0 || std::int64_t(after[i]) != std::int64_t(before[i]) + 1) return false;
            for (int j = 0; j < i; ++j) if (slots[j] == slots[i]) return false;
        }
        if (++accepted_ == 0) ++accepted_; // zero always means no successful evidence
        last_.id = accepted_; last_.rows = rows; last_.nodes = nodes;
        last_.backend_splits = splits; last_.graph_reused = reused;
        for (int i = 0; i < rows; ++i) {
            last_.slots[i] = slots[i]; last_.before[i] = before[i]; last_.after[i] = after[i];
        }
        return true;
    }
};
}
