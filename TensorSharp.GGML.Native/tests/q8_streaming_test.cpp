// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_q8_streaming.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

extern "C" {
const char* TSGgml_GetLastError();
int TSGgml_GetGpuDeviceCount(int type);
int TSGgml_IsBackendAvailable(int type);
int TSGgml_MultiDeviceInit(int type, const int* devices, int count);
int TSGgml_GetCacheMemoryUsage(int rank, std::int64_t* reserved, std::int64_t* committed,
    std::int64_t* budget, std::int64_t* preload_reserved, std::int64_t* preload_committed);
void TSGgml_TestQ8StreamingFailNext(int flags);
void TSGgml_Shutdown();
}
static void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s: %s\n", message, TSGgml_GetLastError()); std::exit(1); }
}
static void contract() {
    require(TSGgml_Q8StreamingPayloadBytes(32, 1, 1) == 768, "small aligned payload");
    require(TSGgml_Q8StreamingPayloadBytes(32, INT32_MAX, 1) == 81604378880LL,
        "row limit must use wide size arithmetic");
    require(TSGgml_Q8StreamingPayloadBytes(31, 1, 1) == 0, "reject unaligned K");
    require(TSGgml_Q8StreamingPayloadBytes(32, 0, 1) == 0, "reject empty tile");
    require(TSGgml_Q8StreamingPayloadBytes(32, 1, 524281) == 0, "reject CUDA grid overflow");
    require(TSGgml_Q8StreamingPayloadBytes(INT64_MAX, 1, 1) == 0, "reject oversized K");
    float input[32]{};
    void* handle = reinterpret_cast<void*>(1);
    require(TSGgml_Q8StreamingCreate(0, 32, 1, 1, input, 767, &handle) == 0 && handle == nullptr,
        "reject insufficient cap before acquiring resources");
    require(TSGgml_Q8StreamingDestroy(nullptr) == 1, "empty release");
    std::puts("Q8 streaming contract passed.");
}
static void check(int inner, int columns, int tile_rows) {
    constexpr int rows = 129;
    const std::size_t row_bytes = std::size_t(inner / 32) * 34;
    std::vector<unsigned char> weights(row_bytes * rows), staging(row_bytes * tile_rows);
    std::vector<float> input(std::size_t(inner) * columns), full(std::size_t(rows) * columns),
        tiled(full.size()), output(std::size_t(tile_rows) * columns);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = float(int((i * 7919 + 43) % 65537) - 32768) / 32768.0f + float(i % 13) * 0x1p-22f;
    void* reference = nullptr;
    void* session = nullptr;
    require(TSGgml_Q8StreamingCreate(0, inner, rows, columns, input.data(),
        TSGgml_Q8StreamingPayloadBytes(inner, rows, columns), &reference) == 1, "create full oracle");
    const auto bounded = TSGgml_Q8StreamingPayloadBytes(inner, tile_rows, columns);
    require(TSGgml_Q8StreamingCreate(0, inner, tile_rows, columns, input.data(), bounded, &session) == 1,
        "create bounded session");
    for (int revision = 0; revision < 2; ++revision) {
        for (int row = 0; row < rows; ++row) for (int block = 0; block < inner / 32; ++block) {
            // Half scale 0x2400 is exactly 1/64. Build published Q8 bytes
            // directly; the independent oracle never calls production decode.
            auto* bytes = weights.data() + std::size_t(row) * row_bytes + block * 34;
            bytes[0] = 0; bytes[1] = 0x24;
            for (int k = 0; k < 32; ++k) bytes[2 + k] = static_cast<unsigned char>(
                static_cast<std::int8_t>(((row * 19 + (block * 32 + k) * 37 + revision * 11) % 256) - 128));
        }
        require(TSGgml_Q8StreamingExecute(reference, weights.data(), rows, full.data()) == 1, "full projection");
        for (int first = 0; first < rows; first += tile_rows) {
            const int count = std::min(tile_rows, rows - first);
            std::memcpy(staging.data(), weights.data() + std::size_t(first) * row_bytes, row_bytes * count);
            require(TSGgml_Q8StreamingExecute(session, staging.data(), count, output.data()) == 1, "tile projection");
            std::fill(staging.begin(), staging.end(), 0xA5); // Host tile is reusable immediately.
            for (int column = 0; column < columns; ++column)
                std::copy_n(output.data() + std::size_t(column) * count, count,
                    tiled.data() + std::size_t(column) * rows + first);
        }
        require(std::memcmp(full.data(), tiled.data(), full.size() * sizeof(float)) == 0,
            "output-row tiling changed result bytes");
        double error_squared = 0, norm = 0;
        for (int column = 0; column < columns; ++column) for (int row = 0; row < rows; ++row) {
            double expected = 0;
            for (int k = 0; k < inner; ++k) {
                const int quant = ((row * 19 + k * 37 + revision * 11) % 256) - 128;
                expected += (double(quant) / 64.0) * input[std::size_t(column) * inner + k];
            }
            const float actual = tiled[std::size_t(column) * rows + row];
            const double error = std::abs(actual - expected);
            require(std::isfinite(actual) && error <= 0.0001 + 0.000006 * std::abs(expected), "independent double oracle");
            error_squared += error * error; norm += expected * expected;
        }
        require(std::sqrt(error_squared / std::max(norm, 1e-300)) <= 0.000004, "independent relative L2");
    }
    require(TSGgml_Q8StreamingDestroy(reference) == 1 && TSGgml_Q8StreamingDestroy(session) == 1, "release sessions");
    std::printf("Q8 streaming K=%d N=%d tile=%d bytes=%lld: exact full/tiled, double oracle, 2 revisions passed\n",
        inner, columns, tile_rows, static_cast<long long>(bounded));
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    contract();
    if (argc == 2 && std::strcmp(argv[1], "--contract") == 0) return 0;
    if (TSGgml_GetGpuDeviceCount(3) < 1) return 77;
    if (argc == 2 && std::strcmp(argv[1], "--singleton") == 0) {
        // Ordinary one-GPU models use this initialization path, which does not
        // populate DeviceState.device_index. Test in its own process because
        // the existing backend cannot initialize again after global shutdown.
        require(TSGgml_IsBackendAvailable(3) == 1, "initialize singleton CUDA");
        check(96, 9, 63);
        TSGgml_Shutdown();
        std::puts("Q8 streaming singleton initialization passed.");
        return 0;
    }
    const int device = 0;
    require(TSGgml_MultiDeviceInit(3, &device, 1) == 1, "initialize CUDA");
    for (int inner : {32, 96, 1024}) for (int columns : {1, 8, 9, 17, 33})
        for (int tile : {1, 63, 64, 65}) check(inner, columns, tile);
    // A failure after physical allocation still transfers an owned handle,
    // and a failed release keeps it valid for a later retry.
    float input[32]{};
    void* handle = nullptr;
    TSGgml_TestQ8StreamingFailNext(3);
    require(TSGgml_Q8StreamingCreate(0, 32, 1, 1, input, 768, &handle) == 0 && handle != nullptr,
        "failed construction must preserve acquired resource ownership");
    require(TSGgml_Q8StreamingDestroy(handle) == 0, "injected release failure");
    require(TSGgml_Q8StreamingDestroy(handle) == 1, "release retry");
    std::int64_t reserved, committed, budget, pre_reserved, pre_committed;
    require(TSGgml_GetCacheMemoryUsage(0, &reserved, &committed, &budget, &pre_reserved, &pre_committed) == 1,
        "query untouched cache accounting");
    require(reserved == 0 && committed == 0 && pre_reserved == 0 && pre_committed == 0,
        "streaming must not enter lazy or preload caches");
    TSGgml_Shutdown();
    std::puts("Q8 streaming: 60 explicit-rank cases x 2 revisions, ownership failure/retry, and no-cache checks passed.");
    return 0;
}
