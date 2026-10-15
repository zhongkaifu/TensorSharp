// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml-backend.h"
#include <atomic>
#include <chrono>
#include <cstdint>
namespace tsg {
inline std::int64_t moe_pipeline_clock_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Diagnostic only. The target is preallocated and lives through stream drain.
// The callback performs one clock read/store, no CUDA calls or allocation.
bool moe_pipeline_stream_mark(ggml_backend_t backend, std::atomic<std::int64_t>* target);
}
