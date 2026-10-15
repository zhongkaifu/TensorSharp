// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "../glm_joined_workers.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

static void require(bool okay, const char * message)
{
    if (!okay) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

int main()
{
    std::atomic<bool> stopped(false);
    std::atomic<int> completed(0);
    tsg_glm::run_joined_workers(4, stopped, [&]() { completed.fetch_add(1); });
    require(completed == 4 && !stopped, "normal workers all joined");

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() / ("glm-workers-" + std::to_string(stamp));
    require(std::filesystem::create_directory(directory), "temporary directory");
    const auto path = directory / "checkpoint.bin";
    struct cleanup
    {
        std::filesystem::path path, directory;
        ~cleanup()
        {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            std::filesystem::remove(directory, ignored);
        }
    } clean{path, directory};
    {
        tsg_glm::unique_file file(std::fopen(path.string().c_str(), "wb"));
        require(file != nullptr, "create temporary checkpoint");
        require(std::fputc(42, file.get()) == 42, "write checkpoint");
    }

    intptr_t file_handle = -1;
    bool caught_allocation = false;
    try
    {
        tsg_glm::run_joined_workers(1, stopped, [&]()
        {
            std::vector<tsg_glm::unique_file> files(1);
            files[0].reset(std::fopen(path.string().c_str(), "rb"));
            if (!files[0]) throw std::runtime_error("cannot open temporary checkpoint");
#if defined(_WIN32)
            file_handle = _get_osfhandle(_fileno(files[0].get()));
#else
            file_handle = fileno(files[0].get());
#endif
            // Simulate staging.resize / strip-slab allocation failure after
            // opening a shard; no actual memory exhaustion is involved.
            throw std::bad_alloc();
        });
    }
    catch (const std::bad_alloc &) { caught_allocation = true; }
    require(caught_allocation && stopped, "worker allocation exception reaches caller");
    require(file_handle != -1, "file opened before exception");
#if defined(_WIN32)
    BY_HANDLE_FILE_INFORMATION info{};
    require(!GetFileInformationByHandle((HANDLE) file_handle, &info) && GetLastError() == ERROR_INVALID_HANDLE,
            "worker exception closes file handle");
#else
    errno = 0;
    require(fcntl((int) file_handle, F_GETFD) == -1 && errno == EBADF, "worker exception closes file handle");
#endif

    stopped = false;
    std::atomic<int> active(0), finished(0);
    int launches = 0;
    bool caught_start = false;
    const auto start = [&](auto work) -> std::thread
    {
        if (++launches == 2)
            throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        return std::thread(work);
    };
    try
    {
        tsg_glm::run_joined_workers(3, stopped, [&]()
        {
            active.fetch_add(1);
            while (!stopped.load()) std::this_thread::yield();
            active.fetch_sub(1);
            finished.fetch_add(1);
        }, start);
    }
    catch (const std::system_error &) { caught_start = true; }
    require(caught_start && launches == 2 && stopped, "partial thread creation failure reaches caller");
    require(active == 0 && finished == 1, "already started worker joined before exception escapes");

    stopped = false;
    std::atomic<int> entered(0), unwound(0);
    bool caught_first = false;
    try
    {
        tsg_glm::run_joined_workers(2, stopped, [&]()
        {
            struct on_exit
            {
                std::atomic<int> & count;
                ~on_exit() { count.fetch_add(1); }
            } leave{unwound};
            const int index = entered.fetch_add(1);
            while (entered.load() < 2) std::this_thread::yield();
            if (index == 0) throw std::bad_alloc();
            // stop is set only after the other exception has been captured.
            while (!stopped.load()) std::this_thread::yield();
            throw std::runtime_error("later worker exception");
        });
    }
    catch (const std::bad_alloc &) { caught_first = true; }
    require(caught_first && entered == 2 && unwound == 2, "first exception preserved and both workers joined");

    stopped = false;
    bool caught_unknown = false;
    try { tsg_glm::run_joined_workers(1, stopped, []() { throw 17; }); }
    catch (int value) { caught_unknown = value == 17; }
    require(caught_unknown && stopped, "non-standard worker exception reaches caller");

    stopped = false;
    completed = 0;
    tsg_glm::run_joined_workers(3, stopped, [&]()
    {
        tsg_glm::unique_file file(std::fopen(path.string().c_str(), "rb"));
        if (!file || std::fgetc(file.get()) != 42) throw std::runtime_error("retry read failed");
        completed.fetch_add(1);
    });
    require(completed == 3 && !stopped, "successful retry after allocation and startup failures");
    std::puts("GLM joined-worker tests passed (normal, allocation, file cleanup, startup, first exception, unknown exception, retry)");
    return 0;
}
