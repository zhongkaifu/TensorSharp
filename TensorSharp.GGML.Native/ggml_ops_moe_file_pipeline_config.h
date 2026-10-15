// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tsg {
struct HostMoeFilePipelineSettings {
    bool enabled = false;
    int minimum_rows = 64;
    std::size_t minimum_payload_bytes = 32u << 20;
    std::size_t maximum_ring_bytes = 16u << 20;
};

// All operations use the enclosing file-MoE compute mutex. Configuration
// itself allocates nothing, and a stale token cannot affect a later owner.
// Begin/end publish only after quiescent physical release succeeds.
class HostMoeFilePipelineConfiguration {
    HostMoeFilePipelineSettings settings_;
    std::uint64_t token_ = 0, next_;
public:
    explicit HostMoeFilePipelineConfiguration(std::uint64_t next = 1) : next_(next) {}
    bool active() const noexcept { return token_ != 0; }
    bool owns(std::uint64_t token) const noexcept { return token != 0 && token == token_; }
    const HostMoeFilePipelineSettings& settings() const noexcept { return settings_; }
    static bool valid(HostMoeFilePipelineSettings value) noexcept {
        return value.minimum_rows > 32 && value.minimum_rows <= (1 << 20)
            && value.minimum_payload_bytes > 0
            && value.maximum_ring_bytes >= 128 && value.maximum_ring_bytes <= (16u << 20)
            && value.maximum_ring_bytes % 128 == 0;
    }
    template<class Release> std::uint64_t begin(HostMoeFilePipelineSettings value, Release release) {
        if (active() || !next_ || !valid(value)) return 0;
        release();
        settings_ = value;
        token_ = next_++;
        return token_;
    }
    template<class Release> bool end(std::uint64_t token, Release release) {
        if (!owns(token)) return false;
        release();
        settings_ = {};
        token_ = 0;
        return true;
    }
};

inline bool host_moe_file_pipeline_eligible(const HostMoeFilePipelineSettings& policy,
    int rows, std::size_t payload_bytes, bool exact_file_sources, bool cuda,
    bool single_device, bool graph_admitted) noexcept {
    return policy.enabled && rows >= policy.minimum_rows && payload_bytes >= policy.minimum_payload_bytes
        && exact_file_sources && cuda && single_device && graph_admitted;
}

inline std::size_t host_moe_file_pipeline_slot_bytes(std::size_t payload_bytes,
    std::size_t maximum_ring_bytes) noexcept {
    return (std::min)({std::size_t(8u << 20), payload_bytes / 2, maximum_ring_bytes / 2}) & ~std::size_t(63);
}
} // namespace tsg
