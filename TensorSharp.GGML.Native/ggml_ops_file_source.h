// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <array>
#include <utility>
#include <atomic>
#include <functional>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tsg {
// Exact loader-registered file extent. Never infer file offsets from mmap
// addresses: views may start at nonzero offsets or belong to different shards.
class ExpertFileSource {
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE direct_file_ = INVALID_HANDLE_VALUE;
#else
    int file_ = -1;
    int direct_file_ = -1;
#endif
    std::uint64_t offset_ = 0, bytes_ = 0;
    std::size_t direct_alignment_ = 0;
    mutable std::atomic<bool> direct_disabled_{false};
public:
    ExpertFileSource(const char* utf8_path, std::uint64_t offset, std::uint64_t bytes, bool request_direct = false)
        : offset_(offset), bytes_(bytes)
    {
        if (!utf8_path || !*utf8_path || !bytes || offset > UINT64_MAX - bytes)
            throw std::runtime_error("Invalid expert file extent");
#if defined(_WIN32)
        if (!utf8_path || !bytes || offset > UINT64_MAX - bytes) throw std::runtime_error("Invalid expert file extent");
        int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8_path, -1, nullptr, 0);
        if (count <= 1 || count > 32768) throw std::runtime_error("Invalid expert source path");
        std::vector<wchar_t> path(count);
        if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8_path, -1, path.data(), count))
            throw std::runtime_error("Cannot decode expert source path");
        file_ = CreateFileW(path.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open expert file source: " + std::to_string(GetLastError()));
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart < 0
            || offset > static_cast<std::uint64_t>(size.QuadPart)
            || bytes > static_cast<std::uint64_t>(size.QuadPart) - offset) {
            CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Expert file extent exceeds its shard");
        }
#if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0602
        if (request_direct) {
            // FILE_STORAGE_INFO reports the largest storage requirement for a
            // spanned/striped volume. Failure keeps the original buffered path.
            // https://learn.microsoft.com/windows/win32/api/winbase/ns-winbase-file_storage_info
            FILE_STORAGE_INFO storage{};
            FILE_ALIGNMENT_INFO requirement{};
            const bool geometry_known = GetFileInformationByHandleEx(file_, FileStorageInfo, &storage, sizeof(storage))
                && GetFileInformationByHandleEx(file_, FileAlignmentInfo, &requirement, sizeof(requirement));
            if (!geometry_known) {
                const DWORD error = GetLastError();
                if (error != ERROR_INVALID_PARAMETER && error != ERROR_NOT_SUPPORTED && error != ERROR_INVALID_FUNCTION) {
                    CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
                    throw std::runtime_error("Cannot query direct expert source alignment: " + std::to_string(error));
                }
            }
            if (geometry_known) {
                const auto alignment = (std::max)({std::uint64_t(4096),
                    std::uint64_t(storage.LogicalBytesPerSector), std::uint64_t(storage.PhysicalBytesPerSectorForAtomicity),
                    std::uint64_t(storage.PhysicalBytesPerSectorForPerformance),
                    std::uint64_t(storage.FileSystemEffectivePhysicalBytesPerSectorForAtomicity),
                    std::uint64_t(requirement.AlignmentRequirement) + 1});
                if (storage.LogicalBytesPerSector && alignment <= (std::numeric_limits<std::size_t>::max)()
                    && (alignment & (alignment - 1)) == 0) {
                    HANDLE direct = CreateFileW(path.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
                    if (direct == INVALID_HANDLE_VALUE) {
                        const DWORD error = GetLastError();
                        if (error != ERROR_INVALID_PARAMETER && error != ERROR_NOT_SUPPORTED && error != ERROR_INVALID_FUNCTION) {
                            CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
                            throw std::runtime_error("Cannot open direct expert source: " + std::to_string(error));
                        }
                    }
                    if (direct != INVALID_HANDLE_VALUE) {
                        BY_HANDLE_FILE_INFORMATION buffered_identity{}, direct_identity{};
                        // Read-only sharing also prevents write/delete replacement
                        // while these handles live. Verify identity and size too.
                        const bool identity_known = GetFileInformationByHandle(file_, &buffered_identity)
                            && GetFileInformationByHandle(direct, &direct_identity);
                        if (!identity_known) {
                            const DWORD error = GetLastError();
                            if (error != ERROR_INVALID_PARAMETER && error != ERROR_NOT_SUPPORTED && error != ERROR_INVALID_FUNCTION) {
                                CloseHandle(direct); CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
                                throw std::runtime_error("Cannot verify direct expert source identity: " + std::to_string(error));
                            }
                        }
                        const bool same = identity_known && (buffered_identity.nFileIndexHigh || buffered_identity.nFileIndexLow)
                            && buffered_identity.dwVolumeSerialNumber == direct_identity.dwVolumeSerialNumber
                            && buffered_identity.nFileIndexHigh == direct_identity.nFileIndexHigh
                            && buffered_identity.nFileIndexLow == direct_identity.nFileIndexLow
                            && buffered_identity.nFileSizeHigh == direct_identity.nFileSizeHigh
                            && buffered_identity.nFileSizeLow == direct_identity.nFileSizeLow;
                        if (same) { direct_file_ = direct; direct_alignment_ = static_cast<std::size_t>(alignment); }
                        else CloseHandle(direct); // Always retain the original exact buffered source.
                    }
                }
            }
        }
#else
        (void)request_direct; // An older Windows target keeps its buffered source.
#endif
#else
        if (offset + bytes > static_cast<std::uint64_t>((std::numeric_limits<off_t>::max)()))
            throw std::runtime_error("Expert file extent exceeds platform offsets");
        file_ = ::open(utf8_path, O_RDONLY | O_CLOEXEC);
        if (file_ < 0) throw std::runtime_error("Cannot open expert file source: " + std::to_string(errno));
        struct stat info{};
        if (::fstat(file_, &info) || info.st_size < 0 || offset > static_cast<std::uint64_t>(info.st_size)
            || bytes > static_cast<std::uint64_t>(info.st_size) - offset) {
            ::close(file_); file_ = -1;
            throw std::runtime_error("Expert file extent exceeds its shard");
        }
#if defined(__linux__) && defined(O_DIRECT)
        if (request_direct) {
            const auto alignment = static_cast<std::uint64_t>((std::max)(decltype(info.st_blksize)(4096), info.st_blksize));
            if (alignment <= (std::numeric_limits<std::size_t>::max)() && (alignment & (alignment - 1)) == 0) {
                const int direct = ::open(utf8_path, O_RDONLY | O_CLOEXEC | O_DIRECT);
                if (direct < 0) {
                    const int error = errno;
                    if (error != EINVAL && error != EOPNOTSUPP && error != ENOTSUP && error != ENOSYS) {
                        ::close(file_); file_ = -1;
                        throw std::runtime_error("Cannot open direct expert source: " + std::to_string(error));
                    }
                }
                if (direct >= 0) {
                    struct stat direct_identity{};
                    if (::fstat(direct, &direct_identity) != 0) {
                        const int error = errno;
                        ::close(direct); ::close(file_); file_ = -1;
                        throw std::runtime_error("Cannot verify direct expert source identity: " + std::to_string(error));
                    }
                    if (info.st_dev == direct_identity.st_dev
                        && info.st_ino == direct_identity.st_ino && info.st_size == direct_identity.st_size) {
                        direct_file_ = direct; direct_alignment_ = static_cast<std::size_t>(alignment);
                    } else ::close(direct); // A path replacement must never change the registered source.
                }
            }
        }
#else
        (void)request_direct; // macOS/iOS and unsupported POSIX targets stay buffered.
#endif
#endif
    }
    ~ExpertFileSource() {
#if defined(_WIN32)
        if (direct_file_ != INVALID_HANDLE_VALUE) CloseHandle(direct_file_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
        if (direct_file_ >= 0) ::close(direct_file_);
        if (file_ >= 0) ::close(file_);
#endif
    }
    ExpertFileSource(const ExpertFileSource&) = delete;
    ExpertFileSource& operator=(const ExpertFileSource&) = delete;
    std::uint64_t bytes() const { return bytes_; }
    std::uint64_t offset() const { return offset_; }
    std::size_t direct_alignment() const noexcept {
        return direct_disabled_.load(std::memory_order_relaxed) ? 0 : direct_alignment_;
    }
    // False means an unsupported or unaligned direct request: the caller must
    // read the ORIGINAL logical span through read(), not copy padding. Real I/O
    // errors and every short direct read throw. No bounce allocation occurs.
    bool try_read_direct(std::uint64_t relative, void* destination, std::size_t count) const {
        if (relative > bytes_ || count > bytes_ - relative || (!destination && count))
            throw std::runtime_error("Direct expert read exceeds registered extent");
        const auto alignment = direct_alignment();
        if (!count || !alignment || (offset_ + relative) % alignment || count % alignment
            || reinterpret_cast<std::uintptr_t>(destination) % alignment) return false;
#if defined(_WIN32)
        struct Event {
            HANDLE handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            ~Event() { if (handle) CloseHandle(handle); }
        };
        thread_local Event event; // a kernel event only, no per-worker data buffer
        if (!event.handle) throw std::runtime_error("Cannot create direct expert read event");
        if (count > (1u << 30)) throw std::runtime_error("Direct expert read exceeds bounded syscall length");
        const auto position = offset_ + relative;
        OVERLAPPED operation{};
        operation.Offset = static_cast<DWORD>(position); operation.OffsetHigh = static_cast<DWORD>(position >> 32);
        operation.hEvent = event.handle;
        if (!ResetEvent(event.handle)) throw std::runtime_error("Cannot reset direct expert read event");
        DWORD got = 0;
        BOOL ok = ReadFile(direct_file_, destination, static_cast<DWORD>(count), &got, &operation);
        if (!ok && GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(direct_file_, &operation, &got, TRUE);
        if (!ok) {
            const DWORD error = GetLastError();
            if (error == ERROR_INVALID_PARAMETER || error == ERROR_NOT_SUPPORTED || error == ERROR_INVALID_FUNCTION) {
                direct_disabled_.store(true, std::memory_order_relaxed); return false;
            }
            throw std::runtime_error("Direct expert file read failed: " + std::to_string(error));
        }
        if (got != count) throw std::runtime_error("Direct expert file read was truncated");
        return true;
#elif defined(__linux__) && defined(O_DIRECT)
        if (count > (std::size_t(1) << 30)) throw std::runtime_error("Direct expert read exceeds bounded syscall length");
        ssize_t got;
        do { got = ::pread(direct_file_, destination, count, static_cast<off_t>(offset_ + relative)); } while (got < 0 && errno == EINTR);
        if (got < 0) {
            const int error = errno;
            if (error == EINVAL || error == EOPNOTSUPP || error == ENOTSUP) {
                direct_disabled_.store(true, std::memory_order_relaxed); return false;
            }
            throw std::runtime_error("Direct expert file read failed: " + std::to_string(error));
        }
        if (static_cast<std::size_t>(got) != count) throw std::runtime_error("Direct expert file read was truncated");
        return true;
#else
        return false;
#endif
    }
    void read(std::uint64_t relative, void* destination, std::size_t count) const {
        if (relative > bytes_ || count > bytes_ - relative || (!destination && count))
            throw std::runtime_error("Expert file read exceeds registered extent");
#if defined(_WIN32)
        struct Event {
            HANDLE handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            ~Event() { if (handle) CloseHandle(handle); }
        };
        // Each worker has at most one outstanding request. Reuse its event;
        // do not create/close a kernel object for every expert projection.
        thread_local Event event;
        if (!event.handle) throw std::runtime_error("Cannot create expert read event");
        auto* target = static_cast<std::uint8_t*>(destination);
        while (count) {
            const auto position = offset_ + relative;
            OVERLAPPED operation{};
            operation.Offset = static_cast<DWORD>(position);
            operation.OffsetHigh = static_cast<DWORD>(position >> 32);
            operation.hEvent = event.handle;
            ResetEvent(event.handle);
            DWORD got = 0, chunk = static_cast<DWORD>(std::min<std::size_t>(count, 1u << 30));
            BOOL ok = ReadFile(file_, target, chunk, &got, &operation);
            if (!ok && GetLastError() == ERROR_IO_PENDING)
                ok = GetOverlappedResult(file_, &operation, &got, TRUE);
            if (!ok || !got) throw std::runtime_error("Expert file read failed or truncated: " + std::to_string(GetLastError()));
            target += got; relative += got; count -= got;
        }
#else
        auto* target = static_cast<std::uint8_t*>(destination);
        while (count) {
            const auto chunk = std::min<std::size_t>(count, std::size_t(1) << 30);
            const auto got = ::pread(file_, target, chunk, static_cast<off_t>(offset_ + relative));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) throw std::runtime_error("Expert file read failed or truncated: " + std::to_string(errno));
            target += got; relative += static_cast<std::size_t>(got); count -= static_cast<std::size_t>(got);
        }
#endif
    }
};

struct ExpertReadTask {
    const ExpertFileSource* source;
    std::uint64_t offset;
    std::uint8_t* destination;
    std::size_t bytes;
    std::uint64_t direct_offset = 0;
    std::uint8_t* direct_destination = nullptr;
    std::size_t direct_bytes = 0;
    bool direct_completed = false;
    void prepare() {
        if (!source || offset > source->bytes() || bytes > source->bytes() - offset || (!destination && bytes))
            throw std::runtime_error("Expert task logical read exceeds registered extent");
        if (direct_bytes) {
            if (!bytes || offset < direct_offset || offset - direct_offset > direct_bytes
                || bytes > direct_bytes - static_cast<std::size_t>(offset - direct_offset)
                || !direct_destination || reinterpret_cast<std::uintptr_t>(destination) < reinterpret_cast<std::uintptr_t>(direct_destination)
                || reinterpret_cast<std::uintptr_t>(destination) - reinterpret_cast<std::uintptr_t>(direct_destination) != offset - direct_offset)
                throw std::runtime_error("Expert direct task does not contain its exact logical destination");
        }
        direct_completed = direct_bytes && source->try_read_direct(direct_offset, direct_destination, direct_bytes);
        if (!direct_completed) source->read(offset, destination, bytes);
    }
};
void read_expert_files(std::vector<ExpertReadTask>& tasks,
    const std::function<void(std::size_t)>& consume);
}

namespace tsg {
struct HostMoeDirectFileGeometry {
    std::uint64_t offset = 0, bytes = 0;
    std::size_t alignment = 0; // zero means buffered-only
};
struct HostMoeDirectFilePiece {
    int slot = 0, part = 0;
    std::size_t expert_offset = 0, staging_offset = 0, bytes = 0;
    std::uint64_t source_offset = 0;
    // Aligned disk span, entirely inside the registered tensor extent and
    // existing staging window. GPU uploads use only staging_offset/bytes.
    std::uint64_t direct_offset = 0;
    std::size_t direct_staging_offset = 0, direct_bytes = 0;
};

inline bool host_moe_direct_alignment_valid(std::size_t alignment) noexcept {
    return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

// Preserve the original router/part order. Alignment prefix/suffix and staging
// address padding consume the same caller-owned window, never a second arena.
// Boundary pieces that cannot read padding within the registered extent use
// their exact original buffered span. No callback survives a window visit.
template<class Visit>
void host_moe_for_each_direct_read_window(const std::vector<std::pair<int, int>>& misses,
    const std::array<std::size_t, 3>& stride,
    const std::array<HostMoeDirectFileGeometry, 3>& files,
    std::size_t window, std::uintptr_t arena_address, Visit&& visit) {
    if (!window) throw std::invalid_argument("Direct expert read plan requires a positive window");
    if (arena_address > (std::numeric_limits<std::uintptr_t>::max)() - window)
        throw std::overflow_error("Direct expert staging address overflows");
    for (int part = 0; part < 3; ++part) {
        const auto& file = files[part];
        if (!stride[part] || !file.bytes || file.offset > UINT64_MAX - file.bytes
            || (file.alignment && !host_moe_direct_alignment_valid(file.alignment)))
            throw std::invalid_argument("Direct expert source geometry is invalid");
    }
    // Reject the complete block before any I/O, including a later invalid miss.
    std::vector<int> slots; slots.reserve(misses.size());
    for (const auto& miss : misses) {
        if (miss.first < 0 || miss.second < 0) throw std::invalid_argument("Direct expert read has a negative miss");
        slots.push_back(miss.first);
        for (int part = 0; part < 3; ++part) {
            const auto count = files[part].bytes / stride[part];
            if (static_cast<std::uint64_t>(miss.second) >= count)
                throw std::invalid_argument("Direct expert read exceeds its registered projection");
        }
    }
    std::sort(slots.begin(), slots.end());
    if (std::adjacent_find(slots.begin(), slots.end()) != slots.end())
        throw std::invalid_argument("Direct expert read repeats a destination slot");
    std::vector<HostMoeDirectFilePiece> pieces;
    std::size_t used = 0;
    const auto flush = [&] {
        if (pieces.empty()) return;
        visit(pieces, used);
        pieces.clear(); used = 0;
    };
    for (const auto& miss : misses) for (int part = 0; part < 3; ++part) {
        const auto& file = files[part];
        const auto start = static_cast<std::uint64_t>(miss.second) * stride[part];
        for (std::size_t offset = 0; offset < stride[part];) {
            if (used == window) flush();
            const auto remaining = stride[part] - offset;
            HostMoeDirectFilePiece piece{miss.first, part, offset, used, (std::min)(remaining, window - used)};
            const auto logical = start + offset;
            piece.source_offset = logical;
            if (file.alignment) {
                const auto alignment = file.alignment;
                const auto absolute = file.offset + logical;
                const auto begin = absolute & ~(static_cast<std::uint64_t>(alignment) - 1);
                const auto prefix = static_cast<std::size_t>(absolute - begin);
                const auto address = arena_address + used;
                const auto pad = static_cast<std::size_t>((alignment - (address & (alignment - 1))) & (alignment - 1));
                const auto room = window - used;
                if (pad < room && begin >= file.offset) {
                    const auto available = (room - pad) / alignment * alignment;
                    if (available > prefix) {
                        const auto count = (std::min)(remaining, available - prefix);
                        const auto wanted = prefix + count;
                        const auto tail = (alignment - wanted % alignment) % alignment;
                        const auto aligned = wanted + tail; // <= available
                        if (aligned <= file.offset + file.bytes - begin) {
                            piece.staging_offset = used + pad + prefix;
                            piece.bytes = count;
                            piece.direct_offset = begin - file.offset;
                            piece.direct_staging_offset = used + pad;
                            piece.direct_bytes = aligned;
                            used += pad + aligned;
                        }
                    } else if (used != 0) {
                        flush();
                        continue;
                    }
                }
            }
            if (!piece.direct_bytes) used += piece.bytes;
            pieces.push_back(piece);
            offset += piece.bytes;
        }
    }
    flush();
}
} // namespace tsg
