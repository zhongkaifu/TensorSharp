// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_file_source.h"
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <chrono>

namespace {
int checks = 0;
void require(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
template<class F> void rejected(F f, const char* message) {
    bool failed = false; try { f(); } catch (const std::exception&) { failed = true; }
    require(failed, message);
}
}
int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / std::filesystem::u8path(
        u8"tensorsharp-专家-" + std::to_string(stamp) + ".bin");
    try {
        std::array<std::uint8_t, 1031> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i * 13);
        { std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }
        const auto name = path.u8string();
        rejected([&] { tsg::ExpertFileSource s(name.c_str(), 7, 1025); }, "extent past EOF accepted");
        rejected([&] { tsg::ExpertFileSource s(name.c_str(), UINT64_MAX - 1, 8); }, "overflow accepted");
        rejected([&] { tsg::ExpertFileSource s(nullptr, 0, 1); }, "null path accepted");
#if !defined(_WIN32)
        int owned_fd = -1;
#endif
        {
            tsg::ExpertFileSource source(name.c_str(), 7, 1024);
            std::array<std::uint8_t, 1030> output{};
            source.read(0, output.data() + 3, 1024);
            require(std::equal(bytes.begin() + 7, bytes.end(), output.begin() + 3), "exact offset read differs");
            require(output[0] == 0 && output[1029] == 0, "read overwrote guards");
            source.read(1024, nullptr, 0);
            rejected([&] { source.read(1024, output.data(), 1); }, "registered extent overread accepted");
            rejected([&] { source.read(1025, output.data(), 0); }, "empty out-of-extent read accepted");
            rejected([&] { source.read(0, nullptr, 1); }, "null payload accepted");
#if !defined(_WIN32)
            struct stat expected{}; require(::stat(name.c_str(), &expected) == 0, "stat failed");
            for (int fd = 0; fd < 1024; ++fd) {
                struct stat actual{};
                if (::fstat(fd, &actual) == 0 && actual.st_dev == expected.st_dev && actual.st_ino == expected.st_ino) {
                    require(owned_fd == -1, "unexpected second source descriptor"); owned_fd = fd;
                }
            }
            require(owned_fd >= 0 && (::fcntl(owned_fd, F_GETFD) & FD_CLOEXEC), "source descriptor not close-on-exec");
            require(::truncate(name.c_str(), 11) == 0, "truncate fixture failed");
            rejected([&] { source.read(0, output.data(), 1024); }, "actual short file read accepted");
#endif
        }
#if !defined(_WIN32)
        errno = 0;
        require(::fcntl(owned_fd, F_GETFD) == -1 && errno == EBADF, "source descriptor leaked");
#endif
        require(std::filesystem::remove(path), "source handle still prevents cleanup");
#if defined(_WIN32)
        std::printf("PASS: %d Windows file-source checks; POSIX truncate/CLOEXEC not run\n", checks);
#else
        std::printf("PASS: %d POSIX file-source checks including real short-read/CLOEXEC/close\n", checks);
#endif
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        std::error_code ec; std::filesystem::remove(path, ec);
        return 1;
    }
}
