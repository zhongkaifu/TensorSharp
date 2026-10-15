// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tsg {
// File ranges can be disjoint while their compact projection destinations are
// contiguous. Fill the admitted ring slot before publishing one DMA operation;
// the producer must finish every range before its consumer can see the slot.
class HostMoeFilePipelinePlan {
public:
    struct Read { std::size_t source, slot_offset, bytes; };
    struct Transfer { int part; std::size_t destination, bytes, first_read, read_count; };
private:
    const std::size_t slot_bytes_;
    std::vector<Read> reads_;
    std::vector<Transfer> transfers_;
    std::array<std::size_t, 3> covered_{};
    int part_ = -1;
    static std::size_t add(std::size_t a, std::size_t b) {
        if (b > (std::numeric_limits<std::size_t>::max)() - a)
            throw std::overflow_error("Host MoE pipeline range overflow");
        return a + b;
    }
public:
    explicit HostMoeFilePipelinePlan(std::size_t slot_bytes) : slot_bytes_(slot_bytes) {
        if (!slot_bytes) throw std::invalid_argument("Host MoE pipeline slot is empty");
    }
    void append(int part, std::size_t source, std::size_t destination, std::size_t bytes) {
        if (part < 0 || part >= 3 || !bytes || part < part_ || part > part_ + 1
            || destination != covered_[part])
            throw std::invalid_argument("Host MoE pipeline range is missing, overlapping or misordered");
        const auto end = add(destination, bytes);
        add(source, bytes); // Validate the whole file range before modifying its plan.
        part_ = part;
        while (bytes) {
            if (transfers_.empty() || transfers_.back().part != part
                || transfers_.back().bytes == slot_bytes_)
                transfers_.push_back({part, destination, 0, reads_.size(), 0});
            auto& transfer = transfers_.back();
            const auto length = std::min(bytes, slot_bytes_ - transfer.bytes);
            // Adjacent ranges can share their syscall, but a source gap is
            // always retained as a distinct exact registered file read.
            if (transfer.read_count && add(reads_.back().source, reads_.back().bytes) == source) {
                reads_.back().bytes = add(reads_.back().bytes, length);
            } else {
                reads_.push_back({source, transfer.bytes, length});
                ++transfer.read_count;
            }
            transfer.bytes += length;
            source += length; destination += length; bytes -= length;
        }
        covered_[part] = end;
    }
    void validate(const std::array<std::size_t, 3>& expected) const {
        if (part_ != 2 || covered_ != expected || transfers_.empty())
            throw std::invalid_argument("Host MoE pipeline projection coverage is incomplete");
        for (const auto& transfer : transfers_)
            if (!transfer.bytes || transfer.bytes > slot_bytes_ || !transfer.read_count
                || transfer.first_read > reads_.size()
                || transfer.read_count > reads_.size() - transfer.first_read)
                throw std::runtime_error("Host MoE pipeline transfer is invalid");
    }
    const std::vector<Read>& reads() const noexcept { return reads_; }
    const std::vector<Transfer>& transfers() const noexcept { return transfers_; }
    std::size_t gate_transfers() const noexcept {
        return static_cast<std::size_t>(std::count_if(transfers_.begin(), transfers_.end(),
            [](const auto& transfer) { return transfer.part != 2; }));
    }
    std::size_t read_job_count(std::size_t index, std::size_t maximum_read_bytes) const {
        if (index >= transfers_.size() || !maximum_read_bytes)
            throw std::invalid_argument("Invalid host MoE parallel read geometry");
        const auto& transfer = transfers_[index];
        std::size_t count = 0;
        for (std::size_t i = 0; i < transfer.read_count; ++i) {
            const auto bytes = reads_[transfer.first_read + i].bytes;
            count = add(count, bytes / maximum_read_bytes + (bytes % maximum_read_bytes != 0));
        }
        return count;
    }
    template<class BeforeRead, class ReadSource>
    void read_job(std::size_t index, std::size_t job, std::size_t maximum_read_bytes,
        std::uint8_t* slot, BeforeRead before_read, ReadSource read) const {
        if (index >= transfers_.size() || !maximum_read_bytes || !slot)
            throw std::invalid_argument("Invalid host MoE parallel read destination");
        const auto& transfer = transfers_[index];
        for (std::size_t i = 0; i < transfer.read_count; ++i) {
            const auto& range = reads_[transfer.first_read + i];
            const auto count = range.bytes / maximum_read_bytes + (range.bytes % maximum_read_bytes != 0);
            if (job < count) {
                const auto offset = job * maximum_read_bytes; // job lies inside this exact source range.
                before_read();
                read(transfer.part, range.source + offset, slot + range.slot_offset + offset,
                    std::min(maximum_read_bytes, range.bytes - offset));
                return;
            }
            job -= count;
        }
        throw std::invalid_argument("Host MoE parallel read index is outside its transfer");
    }
    template<class BeforeRead, class ReadSource>
    void read_transfer(std::size_t index, std::uint8_t* slot, BeforeRead before_read, ReadSource read) const {
        if (index >= transfers_.size() || !slot)
            throw std::invalid_argument("Host MoE pipeline read destination is invalid");
        const auto& transfer = transfers_[index];
        for (std::size_t i = 0; i < transfer.read_count; ++i) {
            const auto& range = reads_[transfer.first_read + i];
            before_read(); // Mid-slot failure/cancellation prevents publication.
            read(transfer.part, range.source, slot + range.slot_offset, range.bytes);
        }
    }
};
} // namespace tsg
