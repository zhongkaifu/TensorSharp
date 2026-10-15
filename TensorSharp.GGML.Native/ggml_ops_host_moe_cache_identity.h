// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <array>
#include <cstdint>

namespace tsg {
// A compact CUDA table may contain experts from different immutable layer
// sources. Equal dimensions, tensor types or local expert IDs are insufficient
// to identify its bytes. Publish this key only after all three uploads finish.
struct ExpertSourceIdentity {
    std::array<const void*, 3> source{};
    std::int32_t expert = -1;
    bool matches(const std::array<const void*, 3>& candidate, std::int32_t id) const noexcept {
        return expert >= 0 && expert == id && source == candidate;
    }
    bool references(const void* pointer) const noexcept {
        return expert >= 0 && (source[0] == pointer || source[1] == pointer || source[2] == pointer);
    }
    void invalidate() noexcept { expert = -1; source = {}; }
    void publish(const std::array<const void*, 3>& candidate, std::int32_t id) noexcept {
        source = candidate; expert = id;
    }
};
}
