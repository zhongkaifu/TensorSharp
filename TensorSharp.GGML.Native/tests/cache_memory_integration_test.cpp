// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    TSGgml_Shutdown();
}
