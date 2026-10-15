// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>

namespace tsg {
// The arena joins authoritative state produced by a nonempty prefill. A fresh
// position-zero slot otherwise skips the [0, position) join and would advance
// zero-initialized recurrence instead of the holder's supplied state.
inline bool qwen4exp_arena_position_valid(int position, int capacity, int rope_position) noexcept {
    return position > 0 && position < capacity && rope_position >= 0;
}

// Protected by the arena mutex; maintenance requires the caller's model gate.
// A failed flush retains both ownership and the configured entry bound.
class Qwen4ExpArenaScope {
    std::uint64_t token_ = 0, next_;
    int entries_ = 4;
public:
    explicit Qwen4ExpArenaScope(std::uint64_t next = 1) noexcept : next_(next) {}
    int entries() const noexcept { return entries_; }
    bool active() const noexcept { return token_ != 0; }
    bool owns(std::uint64_t token) const noexcept { return token && token == token_; }
    std::uint64_t begin(int entries, bool empty) noexcept {
        if (token_ || !next_ || !empty || entries < 1 || entries > 4) return 0;
        token_ = next_++; entries_ = entries; return token_;
    }
    template<class Retire> bool trim(std::uint64_t token, Retire retire) {
        if (!owns(token)) return false;
        retire(); return true;
    }
    template<class Retire> bool end(std::uint64_t token, Retire retire) {
        if (!trim(token, retire)) return false;
        entries_ = 4; token_ = 0; return true;
    }
};
}
