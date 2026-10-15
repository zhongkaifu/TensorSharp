// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "../glm_file_strip.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

static void require(bool pass, const char * name)
{
    if (!pass) { std::fprintf(stderr, "FAIL: %s\n", name); std::exit(1); }
}

int main()
{
    // Windows tmpfile() may try the drive root instead of the writable temp
    // directory. Create an exclusive test directory and remove just our files.
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() / ("glm-strip-" + std::to_string(stamp));
    require(std::filesystem::create_directory(directory), "temporary directory");
    const auto path = directory / "checkpoint.bin";
    struct cleanup
    {
        std::filesystem::path file, directory;
        ~cleanup() { std::filesystem::remove(file); std::filesystem::remove(directory); }
    } remove_test{path, directory};
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path.string().c_str(), "w+b"), &std::fclose);
    require(file != nullptr, "temporary checkpoint");
    std::vector<uint8_t> source(3 * 1024 * 1024 + 37);
    for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t((i * 131 + i / 17) % 251);
    require(std::fwrite(source.data(), 1, source.size(), file.get()) == source.size(), "write checkpoint");
    require(std::fflush(file.get()) == 0, "flush checkpoint");
    size_t file_bytes = 123;
    require(tsg_glm::read_file_size(file.get(), file_bytes) && file_bytes == source.size(), "file size succeeds");
    bool tell_called = false;
    const auto failed_size_seek = [](FILE *) { return -1; };
    const auto observe_tell = [&](FILE *) { tell_called = true; return int64_t(123); };
    require(!tsg_glm::read_file_size(file.get(), file_bytes, failed_size_seek, observe_tell) && file_bytes == 0,
            "file size seek failure refuses and clears result");
    require(!tell_called, "failed size seek does not use old cursor");
    const auto failed_tell = [](FILE *) { return int64_t(-1); };
    require(!tsg_glm::read_file_size(file.get(), file_bytes, tsg_glm::seek_file_end{}, failed_tell) && file_bytes == 0,
            "file size tell failure refuses and clears result");
    require(!tsg_glm::read_file_size(nullptr, file_bytes), "null file size refuses");
    require(tsg_glm::read_file_size(file.get(), file_bytes) && file_bytes == source.size(), "file size retry succeeds");
    std::vector<uint8_t> scratch;
    const auto compare = [&](size_t offset, size_t stride, size_t strip, size_t rows, size_t slab)
    {
        std::vector<uint8_t> output(rows * strip, 0);
        require(tsg_glm::read_file_strips(file.get(), offset, stride, strip, output.size(), output.data(), scratch, slab),
                "gather succeeds");
        for (size_t r = 0; r < rows; ++r)
            require(std::memcmp(output.data() + r * strip, source.data() + offset + r * stride, strip) == 0,
                    "gather preserves every strip byte");
    };
    compare(19, 126, 63, 20000, 1024 * 1024); // Rank offset, multiple slabs and short final slab.
    compare(23, 45, 9, 117, 99);              // Five ranks; only three strips fit each slab.
    compare(7, 37, 37, 110, 100);            // Contiguous rows still work.
    compare(31, 4096, 1024, 100, 16);        // A strip larger than the target slab.
    compare(source.size() - 17 - 5 * 43, 43, 17, 6, 101); // Exact EOF, no trailing row padding.

    uint8_t output[128] = {};
    require(!tsg_glm::read_file_strips(file.get(), source.size() - 10, 23, 11, 22, output, scratch), "short read refuses");
    std::clearerr(file.get());
    require(!tsg_glm::read_file_strips(file.get(), 0, 10, 11, 22, output, scratch), "invalid stride refuses");
    require(!tsg_glm::read_file_strips(file.get(), 0, 10, 0, 22, output, scratch), "zero strip refuses");
    require(!tsg_glm::read_file_strips(file.get(), 0, 10, 7, 22, output, scratch), "partial output row refuses");
    require(!tsg_glm::read_file_strips(file.get(), UINT64_MAX - 3, 10, 7, 21, output, scratch), "offset overflow refuses");
    require(!tsg_glm::read_file_strips(file.get(), INT64_MAX - 20, 10, 7, 21, output, scratch), "span overflow refuses");
    std::clearerr(file.get());
    require(tsg_glm::read_file_range(file.get(), 13, 37, output), "contiguous range succeeds");
    require(std::memcmp(output, source.data() + 13, 37) == 0, "contiguous range preserves bytes");
    require(!tsg_glm::read_file_range(file.get(), source.size() - 3, 4, output), "contiguous short read refuses");
    std::clearerr(file.get());
    require(!tsg_glm::read_file_range(file.get(), INT64_MAX - 1, 4, output), "contiguous span overflow refuses");
    const auto failed_seek = [](FILE *, int64_t) { return -1; };
    {
        require(std::fseek(file.get(), 0, SEEK_SET) == 0, "rewind old-cursor fixture");
        require(failed_seek(file.get(), 1) != 0, "injected seek fails");
        // Demonstrates why merely checking fread, as the old uploader did,
        // can accept the wrong byte after a failed seek.
        require(std::fread(output, 1, 1, file.get()) == 1 && output[0] == source[0] && source[0] != source[1],
                "unchecked failed seek can read the wrong old-cursor byte");
    }
    {
        require(std::fseek(file.get(), 0, SEEK_SET) == 0, "rewind fail-closed fixture");
        output[0] = 0xa5;
        require(!tsg_glm::read_file_range(file.get(), 1, 1, output, failed_seek), "failed seek refuses before reading");
        require(output[0] == 0xa5, "failed seek does not overwrite destination");
        require(std::fgetc(file.get()) == source[0], "failed seek does not consume old-cursor bytes");
    }
    std::puts("GLM checkpoint strip/range/size tests passed (normal, short read, overflow, failed seek/tell, retry)");
    return 0;
}
