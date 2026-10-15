// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tsg_glm_host {

#if !defined(_WIN32)
inline int open_readonly_cloexec(const char * path) {
    return ::open(path, O_RDONLY | O_CLOEXEC);
}
#endif

// Own the view independently of the source descriptor and ggml's non-owning
// CPU buffer wrapper. Destroy the wrapper before closing this mapping.
class readonly_file_mapping {
    void * data_ = nullptr;
    size_t size_ = 0;

public:
    readonly_file_mapping() = default;
    readonly_file_mapping(const readonly_file_mapping &) = delete;
    readonly_file_mapping & operator=(const readonly_file_mapping &) = delete;
    ~readonly_file_mapping() {
        if (!close()) std::fprintf(stderr, "[glm] failed to release host weight mapping\n");
    }
    void * data() const { return data_; }
    size_t size() const { return size_; }

    // A quiescent owner may request release of this read-only view's resident
    // pages while keeping its address, file lifetime and contents unchanged.
    // This is an OS hint, not a released-byte measurement or allocation credit.
    bool trim_residency() noexcept {
        if (!data_ || size_ == 0) return true;
#if defined(_WIN32)
        // No mapping owned here is VirtualLock'ed. The documented unlocked-page
        // path removes these exact pages from the process working set and
        // returns ERROR_NOT_LOCKED. It does not unlock unrelated allocations.
        SetLastError(ERROR_SUCCESS);
        return VirtualUnlock(data_, size_) != 0 || GetLastError() == ERROR_NOT_LOCKED;
#elif defined(MADV_DONTNEED)
        return madvise(data_, size_, MADV_DONTNEED) == 0;
#else
        return false;
#endif
    }

    bool close() noexcept {
        if (!data_) return true;
#if defined(_WIN32)
        if (!UnmapViewOfFile(data_)) return false;
#else
        if (munmap(data_, size_) != 0) return false;
#endif
        data_ = nullptr;
        size_ = 0;
        return true;
    }

    bool open(const std::string & path, std::string & error) {
        error.clear();
        if (data_) { error = "mapping already open"; return false; }
#if defined(_WIN32)
        if (path.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
            error = "UTF-8 path is too long"; return false;
        }
        const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, nullptr, 0);
        if (!length) { error = "invalid UTF-8 file path"; return false; }
        std::wstring wide(static_cast<size_t>(length), L'\0');
        if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, &wide[0], length)) {
            error = "UTF-8 file path conversion failed"; return false;
        }
        struct handle_owner {
            HANDLE value;
            ~handle_owner() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
        };
        // Share reads/deletion (including rename), but prohibit mutation while
        // the checkpoint is being opened. The mapped view keeps the file alive
        // after both source and section handles have been closed.
        handle_owner file{CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                                     nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)};
        if (file.value == INVALID_HANDLE_VALUE) {
            error = "CreateFileW failed: " + std::to_string(GetLastError()); return false;
        }
        LARGE_INTEGER bytes;
        if (!GetFileSizeEx(file.value, &bytes) || bytes.QuadPart <= 0 ||
            static_cast<uint64_t>(bytes.QuadPart) > (std::numeric_limits<size_t>::max)()) {
            error = "empty, unavailable, or unaddressable file size"; return false;
        }
        handle_owner section{CreateFileMappingW(file.value, nullptr, PAGE_READONLY, 0, 0, nullptr)};
        if (!section.value) {
            error = "CreateFileMappingW failed: " + std::to_string(GetLastError()); return false;
        }
        void * view = MapViewOfFile(section.value, FILE_MAP_READ, 0, 0, 0);
        if (!view) { error = "MapViewOfFile failed: " + std::to_string(GetLastError()); return false; }
        data_ = view;
        size_ = static_cast<size_t>(bytes.QuadPart);
#else
        const int fd = open_readonly_cloexec(path.c_str());
        if (fd < 0) { error = std::strerror(errno); return false; }
        struct stat status;
        if (fstat(fd, &status) != 0 || status.st_size <= 0 ||
            static_cast<uint64_t>(status.st_size) > (std::numeric_limits<size_t>::max)()) {
            ::close(fd); error = "empty, unavailable, or unaddressable file size"; return false;
        }
        void * view = mmap(nullptr, static_cast<size_t>(status.st_size), PROT_READ, MAP_SHARED, fd, 0);
        const int map_error = errno;
        ::close(fd);
        if (view == MAP_FAILED) { error = std::strerror(map_error); return false; }
        data_ = view;
        size_ = static_cast<size_t>(status.st_size);
#endif
        return true;
    }
};

// Unknown availability must never authorize a scan of a very large mapping.
// Use subtraction so an adversarial checkpoint size cannot wrap the test.
inline bool can_prefault(uint64_t mapped, uint64_t available, uint64_t headroom) {
    return available > headroom && mapped <= available - headroom;
}

// Demand-read each covered page without allocating a private weight copy. The
// caller must first check the entire warming set against available memory.
// File ranges need not be page aligned: include the final partial page, and
// never read beyond the supplied range. Already-resident pages reuse OS cache.
inline size_t prefault_pages(const void * data, size_t bytes, size_t page_size) {
    if (bytes == 0) return 0;
    if (!data || page_size == 0) throw std::invalid_argument("invalid mapped prefault range");
    const volatile unsigned char * p = static_cast<const volatile unsigned char *>(data);
    (void) p[0];
    size_t pages = 1;
    const size_t first_boundary = page_size - reinterpret_cast<uintptr_t>(data) % page_size;
    for (size_t offset = first_boundary; offset < bytes;) {
        (void) p[offset];
        ++pages;
        if (bytes - offset <= page_size) break;
        offset += page_size;
    }
    return pages;
}

inline uint64_t cap_available(uint64_t available, uint64_t limit, uint64_t current) {
    return (std::min)(available, current >= limit ? uint64_t(0) : limit - current);
}

struct close_text_file {
    void operator()(FILE * file) const noexcept { if (file) std::fclose(file); }
};
// Ownership transfers on entry, including null/read/size failure and an
// allocation exception while growing text.
inline bool read_text_owned(FILE * file, std::string & text) {
    std::unique_ptr<FILE, close_text_file> owner(file);
    if (!file) return false;
    text.clear();
    char block[4096];
    size_t count;
    while ((count = std::fread(block, 1, sizeof(block), file)) != 0) {
        text.append(block, count);
        if (text.size() > 1024 * 1024) return false;
    }
    return std::ferror(file) == 0;
}

inline bool read_text(const char * path, std::string & text) {
    return read_text_owned(std::fopen(path, "r"), text);
}

inline bool parse_counter(const std::string & text, uint64_t & value) {
    std::istringstream words(text);
    std::string number_text, extra;
    if (!(words >> number_text) || (words >> extra) || number_text.empty() ||
        number_text[0] < '0' || number_text[0] > '9') return false;
    char * end = nullptr;
    errno = 0;
    const unsigned long long number = std::strtoull(number_text.c_str(), &end, 10);
    if (errno || !end || *end) return false;
    value = static_cast<uint64_t>(number);
    return true;
}

inline bool path_is_within(const std::string & path, const std::string & root) {
    return path == root || (root == "/" && !path.empty() && path[0] == '/') ||
        (path.size() > root.size() && path.compare(0, root.size(), root) == 0 && path[root.size()] == '/');
}

inline bool valid_proc_path(const std::string & path) {
    if (path.empty() || path[0] != '/' || path.find('\0') != std::string::npos) return false;
    if (path.size() > 1 && path.back() == '/') return false;
    std::istringstream parts(path.substr(1));
    std::string part;
    while (std::getline(parts, part, '/')) if (part.empty() || part == "." || part == "..") return false;
    return true;
}

inline std::string unescape_mount_path(const std::string & text) {
    std::string result;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\\') {
            if (i + 3 >= text.size() || text[i + 1] < '0' || text[i + 1] > '7' ||
                text[i + 2] < '0' || text[i + 2] > '7' || text[i + 3] < '0' || text[i + 3] > '7') return {};
            result.push_back(static_cast<char>((text[i + 1] - '0') * 64 + (text[i + 2] - '0') * 8 + text[i + 3] - '0'));
            i += 3;
        } else result.push_back(text[i]);
    }
    return result;
}

using text_reader = std::function<bool(const std::string &, std::string &)>;

// Resolve the current membership and every visible ancestor, including tighter
// parent quotas. Missing/malformed observation disables warming only; loading
// still uses reclaimable mapped pages. Proc fixtures exercise this on Windows.
inline uint64_t cgroup_available(uint64_t available, const std::string & membership,
                                 const std::string & mounts, const text_reader & read) {
    std::string v1_path, v2_path, line;
    std::istringstream groups(membership);
    while (std::getline(groups, line)) {
        const size_t first = line.find(':'), second = line.find(':', first == std::string::npos ? 0 : first + 1);
        if (first == std::string::npos || second == std::string::npos) return 0;
        const std::string controllers = line.substr(first + 1, second - first - 1);
        const std::string path = line.substr(second + 1);
        if (line.substr(0, first) == "0" && controllers.empty()) v2_path = path;
        if (("," + controllers + ",").find(",memory,") != std::string::npos) v1_path = path;
    }
    const bool v2 = v1_path.empty();
    const std::string member = v2 ? v2_path : v1_path;
    if (!valid_proc_path(member)) return 0;
    std::string mount_point, mount_root;
    std::istringstream mount_lines(mounts);
    while (std::getline(mount_lines, line)) {
        const size_t separator = line.find(" - ");
        if (separator == std::string::npos) continue;
        std::istringstream tail(line.substr(separator + 3));
        std::string type, source, options;
        if (!(tail >> type >> source >> options) || type != (v2 ? "cgroup2" : "cgroup")) continue;
        if (!v2 && ("," + options + ",").find(",memory,") == std::string::npos) continue;
        std::istringstream fields(line.substr(0, separator));
        std::string id, parent, device, root, point;
        if (!(fields >> id >> parent >> device >> root >> point)) return 0;
        root = unescape_mount_path(root);
        point = unescape_mount_path(point);
        if (!valid_proc_path(root) || !valid_proc_path(point)) return 0;
        if (!path_is_within(member, root)) continue;
        // Prefer the mount exposing the greatest number of ancestors.
        if (mount_root.empty() || root.size() < mount_root.size()) { mount_root = root; mount_point = point; }
    }
    if (mount_point.empty()) return 0;
    const std::string suffix = member == mount_root ? "" : member.substr(mount_root == "/" ? 0 : mount_root.size());
    std::string current_path = mount_point == "/" && !suffix.empty() ? suffix : mount_point + suffix;
    for (size_t depth = 0; depth < 1024; ++depth) {
        std::string maximum, usage;
        const std::string prefix = current_path == "/" ? "" : current_path;
        if (!read(prefix + (v2 ? "/memory.max" : "/memory.limit_in_bytes"), maximum)) return 0;
        std::istringstream max_words(maximum);
        std::string word, extra;
        if (!(max_words >> word) || (max_words >> extra)) return 0;
        if (!(v2 && word == "max")) {
            uint64_t limit = 0, used = 0;
            if (!parse_counter(maximum, limit) ||
                !read(prefix + (v2 ? "/memory.current" : "/memory.usage_in_bytes"), usage) ||
                !parse_counter(usage, used)) return 0;
            available = cap_available(available, limit, used);
        }
        if (current_path == mount_point) return available;
        const size_t slash = current_path.rfind('/');
        current_path = slash == 0 ? "/" : current_path.substr(0, slash);
        if (!path_is_within(current_path, mount_point)) return 0;
    }
    return 0;
}

// Current available physical memory, further limited by unused cgroup credit.
// Total installed RAM is not safe: anonymous allocations and other processes
// may already consume most of it when model loading reaches this stage.
inline uint64_t host_available_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullAvailPhys : 0;
#elif defined(__linux__)
    uint64_t available = 0;
    if (FILE * f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1 &&
                kb <= (std::numeric_limits<uint64_t>::max)() / 1024) {
                available = kb * 1024; break;
            }
        }
        std::fclose(f);
    }
    std::string membership, mounts;
    if (!read_text("/proc/self/cgroup", membership) || !read_text("/proc/self/mountinfo", mounts)) return 0;
    return cgroup_available(available, membership, mounts,
        [](const std::string & path, std::string & text) { return read_text(path.c_str(), text); });
#else
    return 0;
#endif
}

} // namespace tsg_glm_host
