// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace tsg {
// Demand-only page reads from several weight mappings in one pool dispatch.
// Each source stays owned by the caller until this synchronous join returns.
void prefetch_mapped_buffers(
    const std::vector<std::pair<const std::uint8_t*, std::size_t>>& buffers);

// Upload one immutable, already-admitted device-cache payload. Page preparation
// overlaps caller-thread uploads, in windows of at most 4 MiB per read worker.
// No staging payload is allocated or pinned; source pages remain reclaimable.
void upload_prefetched_mapped_buffer(const std::uint8_t* source, std::size_t bytes,
    const std::function<void(std::size_t, std::size_t)>& upload);

using MoePrefetchExpert = std::array<std::pair<const std::uint8_t*, std::size_t>, 3>;
// Consume each fully read projection on the caller thread while workers read
// the remaining gate/up/down sources. Returns time until the last read
// completed, excluding subsequent uploads. Sources stay owned until the join.
std::chrono::steady_clock::duration prefetch_mapped_experts(
    const std::vector<MoePrefetchExpert>& experts,
    const std::function<void(std::size_t, std::size_t)>& consume);
}
