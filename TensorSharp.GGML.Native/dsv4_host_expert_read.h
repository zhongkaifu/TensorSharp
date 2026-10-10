// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "dsv4_file_warm.h"
#include "dsv41_engram_io.h"
#include "ggml_ops_shared_cache_budget.h"
#include <array>

namespace tsg_dsv4 {

inline bool host_expert_read_enabled(const char * value) {
    if (!value || !*value || strcmp(value, "0") == 0) return false;
    if (strcmp(value, "1") == 0) return true;
    throw std::runtime_error("TS_DSV4_HOST_EXPERT_READ must be 0 or 1");
}

// Stop paying for residency probes after a warm streak. This only selects a
// faster execution path: the original mmap and CPU kernels always own reads.
// A process-wide major fault is conservative feedback (another model can also
// trigger it). A changed counter immediately restarts demand preparation.
class expert_read_feedback {
    unsigned clean_ = 0;
    bool hot_ = false;
    long faults_ = 0;
public:
    bool hot() const { return hot_; }
    bool ready() const { return clean_ >= 32; }
    void reset() { clean_ = 0; hot_ = false; }
    void prepared(uint64_t read_bytes) {
        if (read_bytes) reset();
        else if (clean_ < 32) ++clean_;
    }
    void arm(long faults) { if (ready()) { hot_ = true; faults_ = faults; } }
    bool check(long faults) {
        if (hot_ && faults != faults_) reset();
        return hot_;
    }
};

// A best-effort residency hint, never a pin/lease: ggml still reads the original
// pageable mmap if the OS evicts a recently checked expert. Probe every eighth
// decode visit and on every prefill/new expert to bound stale observations while
// avoiding hundreds of mincore calls on an unchanged hot decode working set.
class expert_residency_hints {
    uint64_t step_ = 0;
    std::vector<uint64_t> until_;
public:
    void invalidate() { std::fill(until_.begin(), until_.end(), 0); step_ = 0; }
    std::vector<int32_t> select(const std::vector<int32_t> & ids, size_t experts, bool decode) {
        for (auto id : ids) if (id < 0 || size_t(id) >= experts)
            throw std::runtime_error("Host expert read: invalid routed expert");
        if (until_.size() != experts) { until_.assign(experts, 0); step_ = 0; }
        if (++step_ >= UINT64_MAX - 8) { std::fill(until_.begin(), until_.end(), 0); step_ = 1; }
        std::vector<int32_t> needed;
        for (auto id : ids) if (!decode || until_[size_t(id)] <= step_) needed.push_back(id);
        return needed;
    }
    void mark(const std::vector<int32_t> & ids) {
        for (auto id : ids) until_[size_t(id)] = step_ + 8;
    }
};

// Preserve routing order in the graph; deduplicate only immutable file reads.
inline std::vector<file_warm_range> selected_expert_ranges(
        const std::array<file_warm_range, 3> & projections, size_t experts,
        const std::vector<int32_t> & ids) {
    if (!experts) throw std::runtime_error("Host expert read: empty expert dimension");
    for (const auto & p : projections)
        if (!p.mapped || !p.bytes || p.bytes % experts || p.offset > UINT64_MAX - p.bytes)
            throw std::runtime_error("Host expert read: invalid contiguous projection");
    std::vector<uint8_t> seen(experts, 0);
    for (int32_t id : ids) {
        if (id < 0 || size_t(id) >= experts) throw std::runtime_error("Host expert read: invalid routed expert");
        seen[size_t(id)] = 1;
    }
    std::vector<file_warm_range> ranges;
    for (size_t id = 0; id < experts; ++id) if (seen[id]) {
        for (auto p : projections) {
            const uint64_t stride = p.bytes / experts;
            p.offset += id * stride;
            p.mapped = static_cast<const uint8_t *>(p.mapped) + id * stride;
            p.bytes = stride;
            ranges.push_back(p);
        }
    }
    return ranges;
}

#if defined(__linux__)
// Unlike load-time warming, demand reads reuse descriptors, workers and bounded
// staging for the model lifetime. Opening shards during a decode can invalidate
// FUSE page caches. Only selected projections are read, and resident projections
// avoid both pread and worker dispatch. The original mmap stays pageable.
class host_expert_reader {
    std::vector<int> files_;
    std::vector<uint64_t> sizes_;
    tsg_dsv41::engram_io_pool workers_;
    size_t chunk_;
    // Declared before buffers_: quota is released AFTER their physical storage.
    std::shared_ptr<tsg::SharedCacheCharge> charge_;
    std::vector<std::vector<uint8_t>> buffers_;
    std::vector<bool> busy_;
    std::mutex submission_, slots_;
    std::vector<unsigned char> residency_;
    std::array<char, 256> error_{};
public:
    uint64_t requested = 0, resident = 0, read = 0;
    double elapsed_ms = 0;
    host_expert_reader(const std::vector<std::string> & paths, unsigned threads, size_t budget)
        : workers_(threads), chunk_(std::min<size_t>(4 * 1024 * 1024, budget / threads)),
          buffers_(threads), busy_(threads, false) {
        if (chunk_ < 4096) throw std::runtime_error("Host expert read: insufficient staging budget");
        charge_ = tsg::SharedCacheCharge::reserve(0, 3, threads * chunk_);
        if (!charge_) throw std::runtime_error("Host expert read: shared host staging budget exhausted");
        try {
            for (const auto & path : paths) {
                const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd < 0) throw std::runtime_error("Host expert read: cannot open shard");
                try { files_.push_back(fd); }
                catch (...) { close(fd); throw; }
                const off_t size = lseek(fd, 0, SEEK_END);
                if (size < 0) throw std::runtime_error("Host expert read: cannot size shard");
                sizes_.push_back(uint64_t(size));
            }
            for (auto & buffer : buffers_) buffer.resize(chunk_);
            if (!charge_->commit(staging_bytes())) throw std::runtime_error("Host expert read: shared staging commit refused");
        } catch (...) {
            for (int fd : files_) close(fd);
            throw;
        }
    }
    ~host_expert_reader() { for (int fd : files_) close(fd); }
    host_expert_reader(const host_expert_reader &) = delete;
    host_expert_reader & operator=(const host_expert_reader &) = delete;
    size_t staging_bytes() const { return buffers_.size() * chunk_; }
    unsigned threads() const { return workers_.threads(); }
    const char * error() const { return error_.data(); }
    void fail(const char * message) noexcept {
        if (!error_[0]) std::snprintf(error_.data(), error_.size(), "%s", message);
    }

    // Called by the TensorSharp CPU backend: contain read exceptions here.
    // The executor rejects the entire forward if an I/O error was recorded;
    // no partially read staging buffer is ever used as a model weight.
    void warm(const std::vector<file_warm_range> & ranges) noexcept {
        std::lock_guard<std::mutex> submit(submission_);
        if (error_[0]) return;
        const auto begin = std::chrono::steady_clock::now();
        try {
            for (const auto & r : ranges)
                if (r.file >= files_.size() || !r.mapped || r.offset > sizes_[r.file] ||
                    r.bytes > sizes_[r.file] - r.offset)
                    throw std::runtime_error("Host expert read: range exceeds shard");
            std::vector<file_warm_range> jobs;
            for (const auto & r : ranges) {
                requested += r.bytes;
                if (file_warm_detail::fully_resident(r.mapped, r.bytes, residency_)) {
                    resident += r.bytes;
                    continue;
                }
                for (uint64_t off = 0; off < r.bytes; off += chunk_) {
                    auto job = r;
                    job.offset += off;
                    job.mapped = static_cast<const uint8_t *>(r.mapped) + off;
                    job.bytes = std::min<uint64_t>(chunk_, r.bytes - off);
                    jobs.push_back(job);
                }
            }
            std::atomic<uint64_t> bytes{0};
            workers_.run(jobs.size(), [&](size_t index) {
                const auto & job = jobs[index];
                std::vector<unsigned char> pages;
                if (file_warm_detail::fully_resident(job.mapped, job.bytes, pages)) return;
                size_t slot = 0;
                {
                    std::lock_guard<std::mutex> lock(slots_);
                    while (slot < busy_.size() && busy_[slot]) ++slot;
                    if (slot == busy_.size()) throw std::runtime_error("Host expert read: staging slots exhausted");
                    busy_[slot] = true;
                }
                struct release_slot {
                    host_expert_reader & owner; size_t slot;
                    ~release_slot() { std::lock_guard<std::mutex> lock(owner.slots_); owner.busy_[slot] = false; }
                } release{*this, slot};
                const auto result = read_range_exact(buffers_[slot].data(), size_t(job.bytes), job.offset,
                    [&](uint64_t off, void * dst, size_t length) { return pread_at(files_[job.file], off, dst, length); },
                    [](const exact_read_retry &) {});
                if (!result.ok) throw std::runtime_error("Host expert read: incomplete shard read, errno=" + std::to_string(result.error));
                bytes.fetch_add(result.bytes, std::memory_order_relaxed);
            });
            read += bytes.load();
        } catch (const std::exception & ex) { fail(ex.what()); }
        catch (...) { fail("Host expert read: unknown failure"); }
        elapsed_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    }
};
#endif
} // namespace tsg_dsv4
