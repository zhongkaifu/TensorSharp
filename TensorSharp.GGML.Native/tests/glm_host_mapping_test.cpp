// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "../glm_host_mapping.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <new>
#include <stdexcept>
#include <vector>
#if defined(_WIN32)
#include <winioctl.h>
#include <io.h>
#endif

namespace fs = std::filesystem;
using tsg_glm_host::readonly_file_mapping;

// Armed only after opening the file. Force std::string::append's actual next
// heap allocation to fail, without exhausting memory or adding a loader hook.
static bool fail_next_allocation = false;
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void * operator new(size_t bytes) {
    if (fail_next_allocation) { fail_next_allocation = false; throw std::bad_alloc(); }
    if (void * memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void operator delete(void * memory) noexcept { std::free(memory); }
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void operator delete(void * memory, size_t) noexcept { std::free(memory); }

static void require(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(what);
}
static void open_checked(readonly_file_mapping & mapping, const fs::path & path) {
    std::string error;
    if (!mapping.open(path.u8string(), error)) throw std::runtime_error(error);
}

struct scratch_directory {
    fs::path path;
    scratch_directory() {
        path = fs::temp_directory_path() / fs::u8path("glm-mapping-\xe6\xb5\x8b\xe8\xaf\x95-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~scratch_directory() { std::error_code error; fs::remove_all(path, error); }
};

static void test_policy() {
    constexpr uint64_t GiB = uint64_t(1) << 30;
    using tsg_glm_host::can_prefault;
    using tsg_glm_host::cap_available;
    require(!can_prefault(1, 0, 8 * GiB), "unknown availability must not authorize prefault");
    require(!can_prefault(1, 8 * GiB, 8 * GiB), "headroom consumes the entire allowance");
    require(can_prefault(16 * GiB, 24 * GiB, 8 * GiB), "exact boundary fits");
    require(!can_prefault(16 * GiB + 1, 24 * GiB, 8 * GiB), "one byte beyond boundary is denied");
    require(!can_prefault(UINT64_MAX, UINT64_MAX, 8 * GiB), "large mapping cannot overflow decision");
    require(!can_prefault(87 * GiB, 32 * GiB, 8 * GiB), "GLM fixture must not prefault on 32 GiB host");
    require(cap_available(24 * GiB, 16 * GiB, 12 * GiB) == 4 * GiB, "cgroup current usage constrains free credit");
    require(cap_available(24 * GiB, 16 * GiB, 17 * GiB) == 0, "over-limit usage saturates at zero");
    require(cap_available(2 * GiB, 16 * GiB, 1 * GiB) == 2 * GiB, "physical availability still constrains cgroup credit");
    const uint64_t available = tsg_glm_host::host_available_bytes();
#if defined(_WIN32)
    require(available > 0, "current memory availability should be queryable on this test host");
#endif
    std::printf("available_physical_bytes=%llu\n", static_cast<unsigned long long>(available));
}

static void test_cgroup_observer() {
    using tsg_glm_host::cgroup_available;
    const std::string mounts = "30 25 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n";
    std::map<std::string, std::string> files = {
        {"/sys/fs/cgroup/memory.max", "max\n"},
        {"/sys/fs/cgroup/app/memory.max", "1000\n"},
        {"/sys/fs/cgroup/app/memory.current", "900\n"},
    };
    auto read = [&](const std::string & path, std::string & value) {
        auto it = files.find(path);
        if (it == files.end()) return false;
        value = it->second;
        return true;
    };
    require(cgroup_available(32000, "0::/app\n", mounts, read) == 100, "child limit applies even if mount root is unlimited");
    files["/sys/fs/cgroup/app/worker/memory.max"] = "max\n";
    require(cgroup_available(32000, "0::/app/worker\n", mounts, read) == 100, "unlimited leaf still observes constrained ancestor");
    files["/sys/fs/cgroup/memory.max"] = "2000";
    files["/sys/fs/cgroup/memory.current"] = "1950";
    require(cgroup_available(32000, "0::/app\n", mounts, read) == 50, "tightest visible ancestor wins");
    require(cgroup_available(20, "0::/app\n", mounts, read) == 20, "physical available memory also constrains observer");
    files["/sys/fs/cgroup/memory.max"] = "max";
    files.erase("/sys/fs/cgroup/app/memory.current");
    require(cgroup_available(32000, "0::/app\n", mounts, read) == 0, "missing current usage is unknown, not unlimited");
    files["/sys/fs/cgroup/app/memory.current"] = "900";
    files["/sys/fs/cgroup/app/memory.max"] = "broken";
    require(cgroup_available(32000, "0::/app\n", mounts, read) == 0, "malformed max is unknown");
    files.erase("/sys/fs/cgroup/app/memory.max");
    require(cgroup_available(32000, "0::/app\n", mounts, read) == 0, "missing max is unknown");
    require(cgroup_available(32000, "", mounts, read) == 0, "unknown membership cannot authorize warm");
    require(cgroup_available(32000, "0::/../app\n", mounts, read) == 0, "unresolvable namespace path fails closed");
    require(cgroup_available(32000, "0::/app\n", "", read) == 0, "unknown cgroup mount cannot authorize warm");

    const std::string scoped_mount = "30 25 0:28 /app /restricted\\040mount rw - cgroup2 cgroup rw\n";
    files["/restricted mount/memory.max"] = "500";
    files["/restricted mount/memory.current"] = "450";
    files["/restricted mount/worker/memory.max"] = "max";
    require(cgroup_available(32000, "0::/app/worker\n", scoped_mount, read) == 50, "mount root and escaped path resolved exactly");
    require(cgroup_available(32000, "0::/application\n", scoped_mount, read) == 0, "same-prefix sibling is not a descendant");
    require(cgroup_available(32000, "0::/\n", scoped_mount, read) == 0,
            "unresolved namespace-relative membership conservatively skips warming");

    const std::string v1_mounts = "30 25 0:28 / /memory rw - cgroup cgroup rw,memory\n";
    files["/memory/memory.limit_in_bytes"] = "18446744073709551615";
    files["/memory/memory.usage_in_bytes"] = "2000";
    files["/memory/app/memory.limit_in_bytes"] = "1000";
    files["/memory/app/memory.usage_in_bytes"] = "850";
    require(cgroup_available(32000, "4:cpu,cpuacct:/elsewhere\n5:memory:/app\n", v1_mounts, read) == 150,
            "v1 memory controller uses its actual membership and current usage");
    files["/memory/app/memory.usage_in_bytes"] = "1001";
    require(cgroup_available(32000, "5:memory:/app\n", v1_mounts, read) == 0, "v1 over-limit group has zero prefault allowance");
}

static void test_mapping(const fs::path & directory) {
    const fs::path file = directory / fs::u8path("weights-\xe6\x9d\x83\xe9\x87\x8d.bin");
    std::vector<unsigned char> expected(3 * 4096 + 17);
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<unsigned char>((i * 19 + 7) % 251);
    {
        std::ofstream output(file, std::ios::binary);
        output.write(reinterpret_cast<const char *>(expected.data()), expected.size());
        require(output.good(), "create Unicode checkpoint fixture");
    } // source descriptor is closed before the mapping is used
    std::string error;
    readonly_file_mapping mapping;
#if defined(_WIN32)
    DWORD source_handles_before = 0, source_handles_after = 0;
    require(GetProcessHandleCount(GetCurrentProcess(), &source_handles_before) != 0, "read initial source handle count");
#endif
    open_checked(mapping, file);
#if defined(_WIN32)
    require(GetProcessHandleCount(GetCurrentProcess(), &source_handles_after) != 0, "read live-view source handle count");
    require(source_handles_after == source_handles_before, "file and section handles close while view remains alive");
#endif
    require(mapping.size() == expected.size(), "mapping reports exact size");
    require(std::memcmp(mapping.data(), expected.data(), expected.size()) == 0, "all mapped bytes match fixture");
    require(!mapping.open(file.u8string(), error) && mapping.size() == expected.size(), "reopen failure retains existing view");
#if defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory = {};
    require(VirtualQuery(mapping.data(), &memory, sizeof(memory)) == sizeof(memory), "query mapped protection");
    require(memory.Type == MEM_MAPPED && memory.Protect == PAGE_READONLY, "mapping is physically read-only");
#endif
    const auto * mapped_bytes = static_cast<const unsigned char *>(mapping.data());
    require(tsg_glm_host::prefault_pages(mapped_bytes, expected.size(), 4096) == 4, "unaligned last page is warmed");
    require(tsg_glm_host::prefault_pages(mapped_bytes + 1, 4096, 4096) == 2, "unaligned one-page length covers two pages");
    require(tsg_glm_host::prefault_pages(mapped_bytes + 4095, 1, 4096) == 1, "single byte at page end stays in bounds");
    require(tsg_glm_host::prefault_pages(nullptr, 0, 4096) == 0, "empty warming set does not dereference null");
    require(std::memcmp(mapping.data(), expected.data(), expected.size()) == 0, "prefaulting must not modify weights");
    const void* original_address = mapping.data();
    require(mapping.trim_residency(), "quiescent read-only mapping residency hint accepted");
    require(mapping.data() == original_address && mapping.size() == expected.size(), "trim preserves exact pointer and length");
    require(std::memcmp(mapping.data(), expected.data(), expected.size()) == 0, "trim refaults every original file byte");
    const fs::path renamed = directory / "renamed.bin";
    fs::rename(file, renamed); // Windows requires FILE_SHARE_DELETE on live file handles
    require(std::memcmp(mapping.data(), expected.data(), expected.size()) == 0, "view survives rename and closed source handles");
    require(mapping.close() && mapping.close() && !mapping.data() && mapping.size() == 0, "close is idempotent");
    require(mapping.trim_residency(), "empty mapping trim is idempotent");
    open_checked(mapping, renamed);
    require(mapping.close(), "reopen after close releases cleanly");

#if defined(_WIN32)
    DWORD handles_before = 0, handles_after = 0;
    require(GetProcessHandleCount(GetCurrentProcess(), &handles_before) != 0, "read handle count");
#endif
    for (int i = 0; i < 128; ++i) {
        readonly_file_mapping temporary;
        open_checked(temporary, renamed);
        require(static_cast<unsigned char *>(temporary.data())[expected.size() - 1] == expected.back(), "last byte remains accessible");
    }
#if defined(_WIN32)
    require(GetProcessHandleCount(GetCurrentProcess(), &handles_after) != 0, "read final handle count");
    require(handles_after == handles_before, "repeat map/destruction does not leak file or section handles");
#endif
    require(fs::remove(renamed), "all views and handles released before deleting fixture");
    require(!mapping.open(renamed.u8string(), error) && !error.empty() && !mapping.data(), "missing file fails without an owner");
    const fs::path empty = directory / "empty.bin";
    std::ofstream(empty, std::ios::binary).close();
    require(!mapping.open(empty.u8string(), error) && !mapping.data(), "zero-sized file is rejected");
#if defined(_WIN32)
    require(!mapping.open(std::string(1, static_cast<char>(0xff)), error), "invalid UTF-8 is rejected");
#endif
}

static void test_text_allocation_failure(const fs::path & directory) {
    const fs::path path = directory / "observer.txt";
    const std::string expected(8192, 'x');
    { std::ofstream file(path); file << expected; }
    const auto open_file = [&]() {
#if defined(_WIN32)
        return _wfopen(path.c_str(), L"r");
#else
        return std::fopen(path.c_str(), "r");
#endif
    };
    FILE * file = open_file();
    require(file != nullptr, "open text failure fixture");
#if defined(_WIN32)
    const HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file)));
#else
    const int descriptor = fileno(file);
#endif
    bool caught = false;
    std::string text;
    fail_next_allocation = true;
    try {
        tsg_glm_host::read_text_owned(file, text);
    } catch (const std::bad_alloc &) { caught = true; }
    const bool injected = !fail_next_allocation;
    fail_next_allocation = false;
    require(caught && injected, "actual text append allocation failure reaches caller");
#if defined(_WIN32)
    DWORD flags = 0;
    require(!GetHandleInformation(handle, &flags) && GetLastError() == ERROR_INVALID_HANDLE,
            "text allocation failure closes actual Windows file handle");
#else
    errno = 0;
    require(fcntl(descriptor, F_GETFD) == -1 && errno == EBADF,
            "text allocation failure closes actual POSIX descriptor");
#endif
    require(tsg_glm_host::read_text_owned(open_file(), text) && text == expected,
            "text read retries successfully after allocation failure");
    require(!tsg_glm_host::read_text_owned(nullptr, text), "null text owner refuses");
#if !defined(_WIN32)
    const int fd = tsg_glm_host::open_readonly_cloexec(path.c_str());
    require(fd >= 0, "open source with close-on-exec");
    const int flags = fcntl(fd, F_GETFD);
    const int closed = ::close(fd);
    require(flags >= 0 && (flags & FD_CLOEXEC) != 0 && closed == 0,
            "mapped source descriptor sets atomic close-on-exec");
#endif
    require(fs::remove(path), "text fixture deleted after owner cleanup");
}

static void test_large_sparse_mapping(const fs::path & directory) {
    static_assert(sizeof(size_t) >= 8, "GLM checkpoint mapping requires a 64-bit process");
    const fs::path file = directory / "large-sparse.bin";
    const uint64_t size = (uint64_t(1) << 32) + 4097;
#if defined(_WIN32)
    HANDLE handle = CreateFileW(file.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE,
                                nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    require(handle != INVALID_HANDLE_VALUE, "create sparse fixture");
    DWORD transferred = 0;
    const bool sparse = DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &transferred, nullptr) != 0;
    LARGE_INTEGER end;
    end.QuadPart = static_cast<LONGLONG>(size - 1);
    const unsigned char marker = 193;
    const bool written = sparse && SetFilePointerEx(handle, end, nullptr, FILE_BEGIN) &&
                         WriteFile(handle, &marker, 1, &transferred, nullptr) && transferred == 1;
    CloseHandle(handle);
    require(written, "write sparse fixture without allocating 4 GiB of disk space");
#else
    {
        std::ofstream output(file, std::ios::binary);
        output.seekp(static_cast<std::streamoff>(size - 1));
        output.put(static_cast<char>(193));
        require(output.good(), "create large sparse fixture");
    }
#endif
    readonly_file_mapping mapping;
    open_checked(mapping, file);
    require(mapping.size() == size, "mapping preserves file sizes above 32 bits");
    const auto * bytes = static_cast<const unsigned char *>(mapping.data());
    require(bytes[0] == 0 && bytes[size - 2] == 0 && bytes[size - 1] == 193, "sparse holes and >4 GiB tail read correctly");
    require(mapping.close() && fs::remove(file), "large mapped fixture released");
}

int main() {
    try {
        scratch_directory temporary;
        test_policy();
        test_cgroup_observer();
        test_mapping(temporary.path);
        test_text_allocation_failure(temporary.path);
        test_large_sparse_mapping(temporary.path);
        std::puts("PASS glm host mapping: policy, Unicode, read-only, rename, lifetime, >4 GiB sparse file, allocation cleanup, close-on-exec");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL glm host mapping: %s\n", error.what());
        return 1;
    }
}
