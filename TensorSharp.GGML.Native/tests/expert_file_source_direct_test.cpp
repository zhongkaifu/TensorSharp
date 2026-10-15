// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_file_source.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {
int checks = 0;
void require(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
template<class F> void rejected(F operation, const char* message) {
    bool failed = false; try { operation(); } catch (const std::exception&) { failed = true; }
    require(failed, message);
}
}
int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / std::filesystem::u8path(
        u8"tensorsharp-direct-专家-" + std::to_string(stamp) + ".bin");
    auto old_path = path; old_path += ".original";
    struct Cleanup { const std::filesystem::path& path; const std::filesystem::path& old_path;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(old_path, error); }
    } cleanup{path, old_path};
    try {
        std::vector<std::uint8_t> original(1u << 20);
        for (std::size_t i = 0; i < original.size(); ++i) original[i] = static_cast<std::uint8_t>(i * 17 + i / 101);
        { std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char*>(original.data()), original.size());
            require(bool(file), "cannot create direct fixture"); }
        const auto name = path.u8string();
        bool completed_direct = false;
#if !defined(_WIN32)
        std::vector<int> owned;
#endif
        {
            tsg::ExpertFileSource buffered(name.c_str(), 3, original.size() - 7);
            require(buffered.direct_alignment() == 0, "default source unexpectedly enables direct I/O");
            tsg::ExpertFileSource source(name.c_str(), 3, original.size() - 7, true);
            const auto alignment = source.direct_alignment();
            // Bound the fixture too. Unavailable or larger-than-fixture alignment
            // remains a documented skip, never a claimed direct-I/O pass.
            const auto unit = alignment ? alignment : std::size_t(4096);
            if (unit > original.size() / 8) {
                std::puts("SKIP: direct source requires an alignment larger than this bounded fixture"); return 77;
            }
            std::vector<std::uint8_t> storage(4 * unit + 2, 0xa7);
            const auto address = (reinterpret_cast<std::uintptr_t>(storage.data() + 1) + unit - 1) / unit * unit;
            auto* arena = reinterpret_cast<std::uint8_t*>(address);
            const auto source_absolute = 2 * unit;
            tsg::ExpertReadTask task{&source, source_absolute - 3 + 17, arena + 17, unit - 31};
            task.direct_offset = source_absolute - 3; task.direct_destination = arena; task.direct_bytes = unit;
            task.prepare(); completed_direct = task.direct_completed;
            require(std::equal(original.begin() + source_absolute + 17, original.begin() + source_absolute + unit - 14,
                arena + 17), "padded direct or exact buffered payload differs");
            require(arena[unit] == 0xa7 && storage.front() == 0xa7 && storage.back() == 0xa7, "direct read escaped bounded arena");
            if (!completed_direct) require(arena[0] == 0xa7 && arena[unit - 1] == 0xa7,
                "unsupported direct fallback copied unrequested padding");
            std::fill(arena, arena + 2 * unit, 0xa7);
            tsg::ExpertReadTask unaligned{&source, source_absolute - 3 + 17, arena + 18, 33};
            unaligned.direct_offset = source_absolute - 3; unaligned.direct_destination = arena + 1; unaligned.direct_bytes = unit;
            unaligned.prepare();
            require(!unaligned.direct_completed && arena[17] == 0xa7 && arena[51] == 0xa7
                && std::equal(original.begin() + source_absolute + 17, original.begin() + source_absolute + 50, arena + 18),
                "unaligned fallback expanded logical read or changed bytes");
            rejected([&] { auto invalid = task; invalid.offset = source.bytes(); invalid.prepare(); }, "direct task bypassed logical extent");
            rejected([&] { auto invalid = task; ++invalid.destination; invalid.prepare(); }, "direct task accepted wrong logical destination");
            rejected([&] { source.try_read_direct(source.bytes() - 1, arena, unit); }, "direct span bypassed registered tail extent");
            rejected([&] { source.try_read_direct(0, nullptr, unit); }, "direct read accepted null destination");
            require(!source.try_read_direct(0, arena, 1), "unaligned registered head accepted as direct");
            std::array<std::uint8_t, 35> edge{}; edge.fill(0xa7);
            source.read(0, edge.data() + 1, 33);
            require(edge.front() == 0xa7 && edge.back() == 0xa7
                && std::equal(original.begin() + 3, original.begin() + 36, edge.begin() + 1), "registered head fallback differs");
#if !defined(_WIN32)
            struct stat expected{}; require(::stat(name.c_str(), &expected) == 0, "cannot stat direct fixture");
            for (int fd = 0; fd < 1024; ++fd) {
                struct stat actual{};
                if (::fstat(fd, &actual) == 0 && actual.st_dev == expected.st_dev && actual.st_ino == expected.st_ino) {
                    require((::fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0, "direct/buffered descriptor is not CLOEXEC"); owned.push_back(fd);
                }
            }
            require(owned.size() == (alignment ? 3u : 2u), "unexpected direct/buffered descriptor ownership");
            // Replacing the PATH cannot replace either registered source handle.
            std::filesystem::rename(path, old_path);
            { std::ofstream replacement(path, std::ios::binary); std::vector<char> other(original.size(), 0);
                replacement.write(other.data(), other.size()); require(bool(replacement), "cannot replace fixture path"); }
            task.prepare();
            require(std::equal(original.begin() + source_absolute + 17, original.begin() + source_absolute + unit - 14,
                arena + 17), "path replacement changed registered file identity");
            require(::truncate(old_path.c_str(), static_cast<off_t>(source_absolute + unit / 2)) == 0, "cannot truncate registered inode");
            if (completed_direct) rejected([&] { task.prepare(); }, "real short direct read was silently retried or accepted");
            rejected([&] { source.read(source_absolute - 3, arena, unit); }, "real short buffered read accepted");
#else
            // Both source handles use FILE_SHARE_READ, so live-source path
            // replacement and writes cannot race an unbuffered open/read.
            std::error_code rename_error; std::filesystem::rename(path, old_path, rename_error);
            require(bool(rename_error), "live Windows source allowed path replacement");
            const auto wide = path.wstring();
            HANDLE writer = CreateFileW(wide.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
            const auto writer_error = GetLastError();
            if (writer != INVALID_HANDLE_VALUE) CloseHandle(writer);
            require(writer == INVALID_HANDLE_VALUE && writer_error == ERROR_SHARING_VIOLATION,
                "live Windows source allowed mutating write handle");
#endif
        }
#if !defined(_WIN32)
        for (int fd : owned) { errno = 0; require(::fcntl(fd, F_GETFD) == -1 && errno == EBADF, "direct/buffered descriptor leaked"); }
#endif
        require(std::filesystem::remove(path), "direct source handle survived teardown");
        if (!completed_direct) {
            std::printf("SKIP: %d exact fallback/lifecycle checks passed; this filesystem did not complete direct I/O\n", checks); return 77;
        }
        std::printf("PASS: %d actual direct FileSource checks; default-off, padded/exact logical bytes, unsupported/unaligned fallback, extent/identity/close proof; %s\n",
            checks,
#if defined(_WIN32)
            "Windows immutable shared source (real truncate not available)"
#else
            "POSIX actual short read, path replacement, CLOEXEC and descriptor close"
#endif
        );
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
