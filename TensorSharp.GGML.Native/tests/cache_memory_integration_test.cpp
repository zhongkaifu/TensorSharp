// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../ggml_ops_upload_prefetch.h"

extern "C" {
struct TensorView2DDesc { void* data; int dim0, dim1, stride0, stride1; std::int64_t raw_bytes; };
const char* TSGgml_GetLastError();
int TSGgml_GetGpuDeviceCount(int backend_type);
int TSGgml_MultiDeviceInit(int backend_type, const int* indices, int count);
int TSGgml_SetActiveDevice(int rank);
int TSGgml_AddmmQuantF32(TensorView2DDesc result, TensorView2DDesc input, void* weight,
    int type, std::int64_t ne0, std::int64_t ne1, std::int64_t bytes);
int TSGgml_PreloadQuantizedWeight(void* key, void* host, int type,
    std::int64_t ne0, std::int64_t ne1, std::int64_t bytes);
int TSGgml_TestAbandonCachedWeight(void* key, void* host, int type,
    std::int64_t ne0, std::int64_t ne1, std::int64_t bytes);
int TSGgml_GetCacheMemoryUsage(int rank, std::int64_t* reserved, std::int64_t* committed,
    std::int64_t* budget, std::int64_t* preload_reserved, std::int64_t* preload_committed);
void TSGgml_SetDeviceCopyBudget(std::int64_t bytes);
void TSGgml_InvalidateHostBuffer(void* pointer);
void TSGgml_ClearHostBufferCache();
void TSGgml_Shutdown();
}

static void require(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "%s; native error: %s\n", message, TSGgml_GetLastError());
        std::exit(1);
    }
}
struct Usage { std::int64_t reserved, committed, budget, preload_reserved, preload_committed; };
static Usage usage(int rank)
{
    Usage value{};
    require(TSGgml_GetCacheMemoryUsage(rank, &value.reserved, &value.committed, &value.budget,
        &value.preload_reserved, &value.preload_committed) == 1, "cannot query per-rank cache memory");
    require(value.reserved == 0 && value.preload_reserved == 0, "completed call retained pending allocations");
    return value;
}

int main(int argc, char** argv)
{
    const int ranks = argc == 2 ? std::atoi(argv[1]) : 1;
    if (ranks < 1 || ranks > 2) return 1;
    if (TSGgml_GetGpuDeviceCount(3) < ranks) { std::puts("SKIP CUDA device count is insufficient"); return 77; }
    const int indices[] = {0, 1};
    require(TSGgml_MultiDeviceInit(3, indices, ranks) == 1, "CUDA backend initialization failed");
#if defined(__linux__)
    // Query more than one mincore window, including unaligned endpoints. The
    // untouched anonymous range is cold; writing it makes every page resident.
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const auto mapping_bytes = page * 4098;
    auto* mapping = static_cast<unsigned char*>(mmap(nullptr, mapping_bytes,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    require(mapping != MAP_FAILED, "cannot create residency-policy fixture");
    require(tsg::mapped_upload_has_nonresident_pages(mapping + 1, mapping_bytes - 2),
        "untouched source was incorrectly classified as resident");
    std::memset(mapping, 1, mapping_bytes);
    require(!tsg::mapped_upload_has_nonresident_pages(mapping + 1, mapping_bytes - 2),
        "fully resident source would take the cold-upload pipeline");
    require(madvise(mapping + page * 4097, page, MADV_DONTNEED) == 0, "cannot discard tail fixture page");
    require(tsg::mapped_upload_has_nonresident_pages(mapping + 1, mapping_bytes - 2),
        "residency query missed a cold tail beyond its first window");
    require(munmap(mapping, mapping_bytes) == 0, "cannot release residency-policy fixture");
    require(!tsg::mapped_upload_has_nonresident_pages(nullptr, 1)
        && !tsg::mapped_upload_has_nonresident_pages(reinterpret_cast<void*>(1), 0)
        && !tsg::mapped_upload_has_nonresident_pages(reinterpret_cast<void*>(1), SIZE_MAX),
        "invalid residency hints did not retain the ordinary path");
#endif
    constexpr int width = 128, rows = 128;
    constexpr std::int64_t bytes = width * rows * sizeof(float);
    std::vector<float> first(width * rows), second(width * rows), input(width, 1), output(rows);
    for (int row = 0; row < rows; ++row) for (int col = 0; col < width; ++col) {
        first[row * width + col] = float(row % 4 + 1) / 8;
        second[row * width + col] = float(row % 4 + 5) / 8;
    }
    auto run = [&](std::vector<float>& weights) {
        TensorView2DDesc result{output.data(), 1, rows, rows, 1, rows * sizeof(float)};
        TensorView2DDesc source{input.data(), 1, width, width, 1, width * sizeof(float)};
        require(TSGgml_AddmmQuantF32(result, source, weights.data(), 0, width, rows, bytes) == 1,
            "matvec failed on cache or streaming path");
        for (int row = 0; row < rows; ++row)
            require(output[row] == width * weights[row * width], "cache/streamed matvec differs from exact reference");
    };
    // Determine this backend's actual padded payload once, then release it.
    require(TSGgml_SetActiveDevice(0) == 1, "cannot select rank zero");
    TSGgml_SetDeviceCopyBudget(0);
    run(first);
    const auto one = usage(0).committed;
    require(one >= bytes, "CUDA direct cache path was not exercised");
    TSGgml_ClearHostBufferCache();

    for (int rank = 0; rank < ranks; ++rank) {
        require(TSGgml_SetActiveDevice(rank) == 1, "cannot select cache rank");
        TSGgml_SetDeviceCopyBudget(one);
        run(first);
        require(usage(rank).committed == one, "first weight did not populate one cache allocation");
        run(second); // Budget refusal must preserve the complete numerical result.
        require(usage(rank).committed == one, "streaming fallback exceeded the cache cap");
        require(TSGgml_PreloadQuantizedWeight(second.data(), second.data(), 0, width, rows, bytes) == 1,
            "explicit preload was incorrectly subjected to the lazy-copy quota");
        const auto loaded = usage(rank);
        require(loaded.committed == one && loaded.preload_committed >= bytes,
            "explicit preload accounting was merged with the lazy-copy quota");
        run(second);
        const auto hit = usage(rank);
        require(hit.committed == loaded.committed && hit.preload_committed == loaded.preload_committed,
            "preload hit allocated a second copy");
        std::printf("PASS rank %d: direct=%lld preload=%lld cap=%lld; exact cache/streamed/preloaded matvec\n",
            rank, (long long)hit.committed, (long long)hit.preload_committed, (long long)hit.budget);
    }
    if (ranks == 2) require(usage(0).committed == usage(1).committed, "rank counters are not independent");
    TSGgml_InvalidateHostBuffer(second.data());
    for (int rank = 0; rank < ranks; ++rank)
        require(usage(rank).preload_committed == 0, "preload invalidation retained a physical charge");
    TSGgml_ClearHostBufferCache();
    for (int rank = 0; rank < ranks; ++rank) {
        const auto cleared = usage(rank);
        require(cleared.committed == 0 && cleared.preload_committed == 0, "cache teardown leaked charges");
    }

    // Simulate a graph abandoned between cache lookup and its upload loop.
    // The next graph must see completed F32 and quantized bytes on every rank.
    constexpr int block_size = 32, block_bytes = 34; // Q8_0: FP16 scale + 32 signed bytes.
    std::vector<unsigned char> quantized(rows * (width / block_size) * block_bytes);
    for (int row = 0; row < rows; ++row) for (int block = 0; block < width / block_size; ++block) {
        auto* packed = quantized.data() + (row * (width / block_size) + block) * block_bytes;
        const std::uint16_t scale = 0x3c00; // FP16 1.0
        std::memcpy(packed, &scale, sizeof(scale));
        std::memset(packed + sizeof(scale), row % 4 + 1, block_size);
    }
    // CUDA's quantized matvec also quantizes activations to Q8_1. Input 127
    // gives the exactly representable FP16 scale 1 (input 1 gives rounded
    // FP16(1/127)), keeping the expected integer dot product exact.
    std::vector<float> quantized_input(width, 127);
    int opaque_key = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        require(TSGgml_SetActiveDevice(rank) == 1, "cannot select abandoned-graph rank");
        require(TSGgml_TestAbandonCachedWeight(first.data(), first.data(), 0, width, rows, bytes) == 1,
            "cannot stage abandoned F32 cache attempt");
        run(first);
        TensorView2DDesc result{output.data(), 1, rows, rows, 1, rows * sizeof(float)};
        TensorView2DDesc source{quantized_input.data(), 1, width, width, 1, width * sizeof(float)};
        const auto before_streaming = usage(rank).committed;
        TSGgml_SetDeviceCopyBudget(1); // Refuse the optional quantized cache.
        require(TSGgml_AddmmQuantF32(result, source, quantized.data(), 8, width, rows, quantized.size()) == 1,
            "quantized streaming reference failed");
        require(usage(rank).committed == before_streaming, "quantized reference unexpectedly populated the cache");
        const auto streamed = output;
        for (int row = 0; row < rows; ++row)
            require(streamed[row] == width * 127 * (row % 4 + 1), "quantized streaming fixture differs from exact integer reference");
        TSGgml_SetDeviceCopyBudget(0);
        require(TSGgml_TestAbandonCachedWeight(&opaque_key, quantized.data(), 8, width, rows, quantized.size()) == 1,
            "cannot stage abandoned opaque-key quantized cache attempt");
        require(TSGgml_AddmmQuantF32(result, source, &opaque_key, 8, width, rows, quantized.size()) == 1,
            "quantized abandoned-graph cache hit failed");
        for (int row = 0; row < rows; ++row)
            require(output[row] == streamed[row], "abandoned graph quantized cache differs from exact streamed reference");
        require(usage(rank).committed >= bytes + (std::int64_t)quantized.size(),
            "abandoned-graph checks did not retain both cache allocations");
        std::printf("PASS rank %d: abandoned graph leaves completed F32/Q8_0 cache payloads; opaque host key resolved\n", rank);
    }
    TSGgml_ClearHostBufferCache();
    for (int rank = 0; rank < ranks; ++rank)
        require(usage(rank).committed == 0, "abandoned-graph cache teardown leaked charges");

    // Cross the 64 MiB upload-preparation window, with a partial final chunk.
    // Every row is checked, including the tail, after an abandoned graph has
    // published the completed payload. Run with TS_GGML_UPLOAD_PREFETCH=1 to
    // exercise the bounded reader/upload pipeline on Linux CUDA.
    constexpr int large_width = 2048, large_rows = 8193;
    const std::int64_t large_bytes = std::int64_t(large_width) * large_rows * sizeof(float);
    std::vector<float> large(std::size_t(large_width) * large_rows), large_input(large_width, 1), large_output(large_rows);
    for (int row = 0; row < large_rows; ++row)
        for (int col = 0; col < large_width; ++col)
            large[std::size_t(row) * large_width + col] = float(row + 1) / 8192;
    for (int rank = 0; rank < ranks; ++rank) {
        require(TSGgml_SetActiveDevice(rank) == 1, "cannot select large-upload rank");
        TSGgml_SetDeviceCopyBudget(0);
        require(TSGgml_TestAbandonCachedWeight(large.data(), large.data(), 0,
            large_width, large_rows, large_bytes) == 1, "cannot publish large cache upload");
        const auto populated = usage(rank);
        require(populated.committed >= large_bytes, "large cache payload not accounted");
        TensorView2DDesc result{large_output.data(), 1, large_rows, large_rows, 1, large_rows * sizeof(float)};
        TensorView2DDesc source{large_input.data(), 1, large_width, large_width, 1, large_width * sizeof(float)};
        require(TSGgml_AddmmQuantF32(result, source, large.data(), 0, large_width, large_rows, large_bytes) == 1,
            "large cache replay failed");
        for (int row = 0; row < large_rows; ++row)
            require(large_output[row] == large_width * large[std::size_t(row) * large_width],
                "large cache upload lost a chunk or changed its tail");
        require(usage(rank).committed == populated.committed, "large cache hit allocated a duplicate");
        TSGgml_InvalidateHostBuffer(large.data());
        require(usage(rank).committed == 0, "large cache teardown leaked charges");
        std::printf("PASS rank %d: 64 MiB + 8 KiB upload, every row exact, cache replay and release\n", rank);
    }
#if defined(__linux__)
    // Keep the interior nonresident until the cache copy reads it. Endpoints
    // distinguish a completed cold upload from an untouched device allocation;
    // the row oracle also checks every zero-filled interior page. With the
    // environment unset this exercises automatic cold-source selection.
    constexpr int cold_rows = 2049;
    constexpr std::size_t cold_bytes = std::size_t(large_width) * cold_rows * sizeof(float);
    auto* cold = static_cast<float*>(mmap(nullptr, cold_bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    require(cold != MAP_FAILED, "cannot map cold-upload fixture");
    cold[0] = 1;
    cold[cold_bytes / sizeof(float) - 1] = 2;
    for (int rank = 0; rank < ranks; ++rank) {
        require(madvise(reinterpret_cast<unsigned char*>(cold) + page, cold_bytes - 2 * page,
            MADV_DONTNEED) == 0, "cannot discard cold-upload fixture interior");
        require(tsg::mapped_upload_has_nonresident_pages(cold, cold_bytes), "cold fixture became resident");
        require(TSGgml_SetActiveDevice(rank) == 1, "cannot select cold-upload rank");
        require(TSGgml_TestAbandonCachedWeight(cold, cold, 0, large_width, cold_rows, cold_bytes) == 1,
            "cannot publish cold cache upload");
        require(usage(rank).committed >= cold_bytes, "cold cache payload was not charged");
        std::vector<float> cold_output(cold_rows, -1);
        TensorView2DDesc result{cold_output.data(), 1, cold_rows, cold_rows, 1, cold_rows * sizeof(float)};
        TensorView2DDesc source{large_input.data(), 1, large_width, large_width, 1, large_width * sizeof(float)};
        require(TSGgml_AddmmQuantF32(result, source, cold, 0, large_width, cold_rows, cold_bytes) == 1,
            "cold cache replay failed");
        for (int row = 0; row < cold_rows; ++row)
            require(cold_output[row] == (row == 0 ? 1 : row == cold_rows - 1 ? 2 : 0),
                "cold upload changed endpoints or an interior page");
        TSGgml_InvalidateHostBuffer(cold);
        require(usage(rank).committed == 0, "cold cache teardown leaked charges");
        std::printf("PASS rank %d: nonresident upload, exact endpoints/interior and release\n", rank);
    }
    require(munmap(cold, cold_bytes) == 0, "cannot unmap cold-upload fixture");
#endif
    TSGgml_Shutdown();
}
