// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_q8_streaming.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <vector>

extern "C" {
struct TensorView2DDesc { void* data; int dim0, dim1, stride0, stride1; std::int64_t raw_bytes; };
int TSGgml_AddmmQuantF32(TensorView2DDesc result, TensorView2DDesc input, void* weight,
    int type, std::int64_t ne0, std::int64_t ne1, std::int64_t bytes);
void TSGgml_ClearHostBufferCache();
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
    require(TSGgml_WeightStreamingPayloadBytes(1, 33, 1, 1) == 768, "F16 permits a partial K tile");
    require(TSGgml_WeightStreamingPayloadBytes(8, 33, 1, 1) == 0, "Q8 still requires block-aligned K");
    require(TSGgml_WeightStreamingPayloadBytes(0, 32, 1, 1) == 0, "reject unsupported F32 weight type");
    require(TSGgml_WeightStreamingPayloadBytes(1, INT64_MAX, 1, 1) == 0, "reject oversized F16 K");
    float input[32]{};
    void* handle = reinterpret_cast<void*>(1);
    require(TSGgml_Q8StreamingCreate(0, 32, 1, 1, input, 767, &handle) == 0 && handle == nullptr,
        "reject insufficient cap before acquiring resources");
    require(TSGgml_Q8StreamingDestroy(nullptr) == 1, "empty release");
    require(TSGgml_Q8StreamingUploadInput(nullptr, input, 1) == 0, "reject input upload without a session");
    require(TSGgml_WeightStreamingUploadInput(nullptr, input, 1) == 0, "generic upload without session");
    require(TSGgml_ResidentWeightPayloadBytes(0, 8, 32, 128, 8) == 0, "complete MMQ rejects MMVQ shape");
    require(TSGgml_ResidentWeightPayloadBytes(0, 1, 2560, 128, 16) == 0, "complete cuBLAS rejects MMF shape");
    require(TSGgml_ResidentWeightPayloadBytes(0, 0, 32, 128, 36) == 0, "complete matrix rejects unsupported type");
    handle = reinterpret_cast<void*>(1);
    require(TSGgml_ResidentWeightCreate(0, 8, 32, 128, 8, 0, &handle) == 0 && handle == nullptr,
        "invalid complete shape cannot acquire an owned handle");
    require(TSGgml_ResidentWeightUploadRows(nullptr, input, 0, 1) == 0 &&
        TSGgml_ResidentWeightUploadInput(nullptr, input, 0, 1) == 0 &&
        TSGgml_ResidentWeightProject(nullptr) == 0 &&
        TSGgml_ResidentWeightDownload(nullptr, input, 0, 1, 0, 1) == 0, "complete matrix rejects null handles");
    std::puts("Q8 streaming contract passed.");
}
static void double_oracle(const std::vector<float>& input, const std::vector<float>& actual,
        int inner, int rows, int columns, int revision) {
    double error_squared = 0, norm = 0;
    for (int column = 0; column < columns; ++column) for (int row = 0; row < rows; ++row) {
        double expected = 0;
        for (int k = 0; k < inner; ++k) {
            const int quant = ((row * 19 + k * 37 + revision * 11) % 256) - 128;
            expected += (double(quant) / 64.0) * input[std::size_t(column) * inner + k];
        }
        const float value = actual[std::size_t(column) * rows + row];
        const double error = std::abs(value - expected);
        require(std::isfinite(value) && error <= 0.0001 + 0.000006 * std::abs(expected), "independent double oracle");
        error_squared += error * error; norm += expected * expected;
    }
    require(std::sqrt(error_squared / std::max(norm, 1e-300)) <= 0.000004, "independent relative L2");
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
        double_oracle(input, tiled, inner, rows, columns, revision);
    }
    require(TSGgml_Q8StreamingDestroy(reference) == 1 && TSGgml_Q8StreamingDestroy(session) == 1, "release sessions");
    std::printf("Q8 streaming K=%d N=%d tile=%d bytes=%lld: exact full/tiled, double oracle, 2 revisions passed\n",
        inner, columns, tile_rows, static_cast<long long>(bounded));
}

static void check_input_reuse(int inner, int tile_rows) {
    constexpr int rows = 129, capacity = 33;
    constexpr float sentinel = -98765.0f;
    const std::size_t row_bytes = std::size_t(inner / 32) * 34;
    std::vector<unsigned char> weights(row_bytes * rows), staging(row_bytes * tile_rows);
    std::vector<float> input(std::size_t(inner) * capacity), upload(input.size()),
        full(std::size_t(rows) * capacity), tiled(full.size()), output(std::size_t(tile_rows) * capacity + 8);
    void* session = nullptr;
    const auto bounded = TSGgml_Q8StreamingPayloadBytes(inner, tile_rows, capacity);
    require(TSGgml_Q8StreamingCreate(0, inner, tile_rows, capacity, input.data(), bounded, &session) == 1,
        "create reusable input session");
    int revision = 0;
    for (int columns : {33, 1, 17, 8, 9, 16, 32, 33}) {
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = float(int((i * 7919 + 43 + revision * 997) % 65537) - 32768) / 32768.0f
                + float((i + revision) % 13) * 0x1p-22f;
        for (int row = 0; row < rows; ++row) for (int block = 0; block < inner / 32; ++block) {
            auto* bytes = weights.data() + std::size_t(row) * row_bytes + block * 34;
            bytes[0] = 0; bytes[1] = 0x24;
            for (int k = 0; k < 32; ++k) bytes[2 + k] = static_cast<unsigned char>(
                static_cast<std::int8_t>(((row * 19 + (block * 32 + k) * 37 + revision * 11) % 256) - 128));
        }
        void* fresh = nullptr;
        require(TSGgml_Q8StreamingCreate(0, inner, rows, columns, input.data(),
            TSGgml_Q8StreamingPayloadBytes(inner, rows, columns), &fresh) == 1, "create fresh input control");
        require(TSGgml_Q8StreamingExecute(fresh, weights.data(), rows, full.data()) == 1, "fresh input control");
        require(TSGgml_Q8StreamingDestroy(fresh) == 1, "release fresh input control");
        upload = input;
        require(TSGgml_Q8StreamingUploadInput(session, upload.data(), columns) == 1, "replace input in bounded arena");
        std::fill(upload.begin(), upload.end(), std::numeric_limits<float>::quiet_NaN());
        require(TSGgml_Q8StreamingUploadInput(session, nullptr, columns) == 0, "reject null input without poisoning session");
        require(TSGgml_Q8StreamingUploadInput(session, upload.data(), 0) == 0, "reject zero tokens without poisoning session");
        require(TSGgml_Q8StreamingUploadInput(session, upload.data(), capacity + 1) == 0,
            "reject excess tokens without poisoning session");
        for (int first = 0; first < rows; first += tile_rows) {
            const int count = std::min(tile_rows, rows - first);
            std::copy_n(weights.data() + std::size_t(first) * row_bytes, row_bytes * count, staging.data());
            std::fill(output.begin(), output.end(), sentinel);
            require(TSGgml_Q8StreamingExecute(session, staging.data(), count, output.data()) == 1,
                "projection after input replacement and invalid arguments");
            std::fill(staging.begin(), staging.end(), 0xA5);
            require(std::all_of(output.begin() + std::size_t(count) * columns, output.end(),
                [sentinel](float x) { return x == sentinel; }), "short input/weight tile overwrote output tail");
            for (int column = 0; column < columns; ++column)
                std::copy_n(output.data() + std::size_t(column) * count, count,
                    tiled.data() + std::size_t(column) * rows + first);
        }
        require(std::memcmp(full.data(), tiled.data(), std::size_t(columns) * rows * sizeof(float)) == 0,
            "reused input arena differs from a fresh session");
        double_oracle(input, tiled, inner, rows, columns, revision++);
    }
    TSGgml_TestQ8StreamingFailNext(4);
    require(TSGgml_Q8StreamingUploadInput(session, input.data(), 1) == 0,
        "injected failure after physical input replacement");
    require(TSGgml_Q8StreamingExecute(session, weights.data(), 1, output.data()) == 0 &&
        TSGgml_Q8StreamingUploadInput(session, input.data(), 1) == 0,
        "failed upload session must remain unusable");
    require(TSGgml_Q8StreamingDestroy(session) == 1, "release failed upload session");
    std::printf("Q8 input reuse K=%d tile=%d: 8 token shapes, fresh-session parity, double oracle, guards and failure passed\n",
        inner, tile_rows);
}

static double decode_half(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const double magnitude = exponent == 0 ? std::ldexp(double(fraction), -24)
        : std::ldexp(double(1024 + fraction), exponent - 25);
    return (bits & 0x8000) != 0 ? -magnitude : magnitude;
}

static void check_f16(int inner, int rows, int tile_rows, std::initializer_list<int> shapes) {
    const int capacity = *std::max_element(shapes.begin(), shapes.end());
    constexpr float sentinel = -98765.0f;
    // Includes both signs, fractions, normal and subnormal values. The oracle
    // decodes the published half representation independently of CUDA/ggml.
    constexpr std::uint16_t values[] = {0x0001, 0x8001, 0x0400, 0x3c01, 0xb003, 0x3555, 0x3999, 0xb266};
    double decoded[8];
    for (int i = 0; i < 8; ++i) decoded[i] = decode_half(values[i]);
    std::vector<std::uint16_t> weights(std::size_t(inner) * rows), staging(std::size_t(inner) * tile_rows);
    std::vector<float> input(std::size_t(inner) * capacity), upload(input.size()),
        full(std::size_t(rows) * capacity), tiled(full.size()), output(std::size_t(tile_rows) * capacity + 8);
    void* session = nullptr;
    const auto bounded = TSGgml_WeightStreamingPayloadBytes(1, inner, tile_rows, capacity);
    require(TSGgml_WeightStreamingCreate(0, 1, inner, tile_rows, capacity, input.data(), bounded, &session) == 1,
        "create bounded F16 session");
    int revision = 0;
    for (int columns : shapes) {
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = float(int((i * 7919 + 43 + revision * 997) % 65537) - 32768) / 32768.0f
                + float((i + revision) % 13) * 0x1p-22f;
        for (int row = 0; row < rows; ++row) for (int k = 0; k < inner; ++k)
            weights[std::size_t(row) * inner + k] = values[(row * 19 + k * 37 + revision * 11) % 8];
        void* fresh = nullptr;
        require(TSGgml_WeightStreamingCreate(0, 1, inner, rows, columns, input.data(),
            TSGgml_WeightStreamingPayloadBytes(1, inner, rows, columns), &fresh) == 1, "create fresh F16 control");
        require(TSGgml_WeightStreamingExecute(fresh, weights.data(), rows, full.data()) == 1, "fresh F16 projection");
        require(TSGgml_WeightStreamingDestroy(fresh) == 1, "destroy fresh F16 control");
        upload = input;
        require(TSGgml_WeightStreamingUploadInput(session, upload.data(), columns) == 1, "replace F16 session input");
        std::fill(upload.begin(), upload.end(), std::numeric_limits<float>::quiet_NaN());
        require(TSGgml_WeightStreamingUploadInput(session, nullptr, columns) == 0 &&
            TSGgml_WeightStreamingUploadInput(session, upload.data(), 0) == 0 &&
            TSGgml_WeightStreamingUploadInput(session, upload.data(), capacity + 1) == 0,
            "F16 invalid upload arguments must preserve old input");
        for (int first = 0; first < rows; first += tile_rows) {
            const int count = std::min(tile_rows, rows - first);
            std::copy_n(weights.data() + std::size_t(first) * inner, std::size_t(count) * inner, staging.data());
            std::fill(output.begin(), output.end(), sentinel);
            require(TSGgml_WeightStreamingExecute(session, staging.data(), count, output.data()) == 1, "bounded F16 tile");
            std::fill(staging.begin(), staging.end(), std::uint16_t(0x7e00));
            require(std::all_of(output.begin() + std::size_t(count) * columns, output.end(),
                [sentinel](float x) { return x == sentinel; }), "F16 output tail overwritten");
            for (int column = 0; column < columns; ++column)
                std::copy_n(output.data() + std::size_t(column) * count, count,
                    tiled.data() + std::size_t(column) * rows + first);
        }
        require(std::memcmp(full.data(), tiled.data(), std::size_t(columns) * rows * sizeof(float)) == 0,
            "F16 tiled/reused session differs from fresh full projection");
        double error_squared = 0, norm = 0;
        for (int column = 0; column < columns; ++column) for (int row = 0; row < rows; ++row) {
            double expected = 0;
            for (int k = 0; k < inner; ++k)
                expected += decoded[(row * 19 + k * 37 + revision * 11) % 8] * input[std::size_t(column) * inner + k];
            const float actual = tiled[std::size_t(column) * rows + row];
            const double error = std::abs(actual - expected);
            require(std::isfinite(actual) && error <= 0.0001 + 0.000006 * std::abs(expected), "F16 independent double oracle");
            error_squared += error * error; norm += expected * expected;
        }
        require(std::sqrt(error_squared / std::max(norm, 1e-300)) <= 0.000004, "F16 independent relative L2");
        ++revision;
    }
    TSGgml_TestQ8StreamingFailNext(4);
    require(TSGgml_WeightStreamingUploadInput(session, input.data(), 1) == 0, "F16 real-upload failure hook");
    require(TSGgml_WeightStreamingExecute(session, weights.data(), 1, output.data()) == 0 &&
        TSGgml_WeightStreamingUploadInput(session, input.data(), 1) == 0, "F16 failed upload remains unusable");
    require(TSGgml_WeightStreamingDestroy(session) == 1, "release F16 session after upload failure");
    std::printf("F16 streaming K=%d rows=%d tile=%d: %d token shapes, fresh-session parity, double oracle and guards passed\n",
        inner, rows, tile_rows, revision);
}
// The reference invokes the unchanged ggml resident arithmetic on the complete
// Linear. In particular logical N=36 must not become MMVQ on a 32+4 tail tile.
// This comparison is independent of the streaming kernels and dispatch helpers.
static void check_resident(int type, int inner, int rows, int columns, int tile_rows, int tile_columns) {
    const std::size_t row_bytes = type == 8 ? std::size_t(inner / 32) * 34 : std::size_t(inner) * 2;
    std::vector<unsigned char> weights(row_bytes * rows), staging(row_bytes * tile_rows);
    std::vector<float> input(std::size_t(inner) * columns), uploaded(std::size_t(inner) * tile_columns),
        reference(std::size_t(rows) * columns), actual(reference.size());
    constexpr float sentinel = -9876.5f;
    std::vector<float> output(std::size_t(tile_rows) * tile_columns + 8, sentinel);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = float(int((i * 7919 + 43) % 65537) - 32768) / 32768.0f + float(i % 13) * 0x1p-22f;
    for (int row = 0; row < rows; ++row) for (int k = 0; k < inner; ++k) {
        const int pattern = ((row * 19 + k * 37) % 256) - 128;
        if (type == 8) {
            auto* block = weights.data() + std::size_t(row) * row_bytes + (k / 32) * 34;
            block[0] = 0; block[1] = 0x24; // Q8 scale 1/64.
            block[2 + k % 32] = static_cast<unsigned char>(static_cast<std::int8_t>(pattern));
        } else {
            // Finite normal FP16 values with both signs and varied mantissas.
            const auto half = std::uint16_t((pattern < 0 ? 0x8000 : 0) | 0x3400 | ((std::abs(pattern) * 7) & 1023));
            std::memcpy(weights.data() + std::size_t(row) * row_bytes + k * 2, &half, 2);
        }
    }
    TensorView2DDesc source{input.data(), columns, inner, inner, 1, std::int64_t(input.size() * sizeof(float))};
    TensorView2DDesc result{reference.data(), columns, rows, rows, 1, std::int64_t(reference.size() * sizeof(float))};
    require(TSGgml_AddmmQuantF32(result, source, weights.data(), type, inner, rows, weights.size()) == 1,
        "unchanged ggml resident projection");
    TSGgml_ClearHostBufferCache();
    const auto payload = TSGgml_WeightStreamingPayloadBytesEx(type, inner, tile_rows, tile_columns, 0, 1, columns, rows);
    require(payload > 0, "resident workspace query");
    void* session = nullptr;
    std::copy_n(input.data(), uploaded.size(), uploaded.data());
    require(TSGgml_WeightStreamingCreateEx(0, type, inner, tile_rows, tile_columns, 1, columns, rows,
        uploaded.data(), payload, &session) == 1, "create resident-compatible session");
    for (int first_column = 0; first_column < columns; first_column += tile_columns) {
        const int count_columns = std::min(tile_columns, columns - first_column);
        if (first_column) {
            std::copy_n(input.data() + std::size_t(first_column) * inner, std::size_t(count_columns) * inner, uploaded.data());
            require(TSGgml_WeightStreamingUploadInput(session, uploaded.data(), count_columns) == 1,
                "reuse resident-compatible input with original logical shape");
        }
        std::fill(uploaded.begin(), uploaded.end(), std::numeric_limits<float>::quiet_NaN());
        for (int first_row = 0; first_row < rows; first_row += tile_rows) {
            const int count_rows = std::min(tile_rows, rows - first_row);
            std::memcpy(staging.data(), weights.data() + std::size_t(first_row) * row_bytes, row_bytes * count_rows);
            std::fill(output.begin(), output.end(), sentinel);
            require(TSGgml_WeightStreamingExecute(session, staging.data(), count_rows, output.data()) == 1,
                "resident-compatible tiled projection");
            std::fill(staging.begin(), staging.end(), 0xA5);
            require(std::all_of(output.begin() + std::size_t(count_rows) * count_columns, output.end(),
                [sentinel](float value) { return value == sentinel; }), "resident output canary");
            for (int column = 0; column < count_columns; ++column)
                std::copy_n(output.data() + std::size_t(column) * count_rows, count_rows,
                    actual.data() + std::size_t(first_column + column) * rows + first_row);
        }
    }
    require(TSGgml_WeightStreamingDestroy(session) == 1, "destroy resident-compatible session");
    double error = 0, norm = 0, actual_norm = 0, dot = 0, max_error = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(actual[i]), "resident projection contains a nonfinite value");
        const double delta = double(actual[i]) - reference[i];
        error += delta * delta; norm += double(reference[i]) * reference[i];
        actual_norm += double(actual[i]) * actual[i]; dot += double(actual[i]) * reference[i];
        max_error = std::max(max_error, std::abs(delta));
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-300));
    const double cosine = dot / std::max(std::sqrt(norm * actual_norm), 1e-300);
    std::printf("resident type=%d K=%d rows=%d N=%d tile=%dx%d payload=%lld relL2=%.9g cosine=%.12g maxabs=%.9g\n",
        type, inner, rows, columns, tile_rows, tile_columns, static_cast<long long>(payload), relative, cosine, max_error);
    require(relative <= 0.001 && cosine >= 0.999999, "resident-compatible strict numerical gate");
}

// A single complete logical MMQ/cuBLAS call; host tiling never changes native
// arithmetic. Require bitwise equality, including tail rows and input reuse.
static void check_resident_full(int type, int inner, int rows, int columns, int tile_rows, int tile_columns) {
    const std::size_t row_bytes = type == 8 ? std::size_t(inner / 32) * 34 : std::size_t(inner) * 2;
    std::vector<unsigned char> weights(row_bytes * rows), staging(row_bytes * tile_rows);
    std::vector<float> input(std::size_t(inner) * columns), uploaded(std::size_t(inner) * tile_columns),
        reference(std::size_t(rows) * columns), actual(reference.size());
    constexpr float sentinel = -9876.5f;
    std::vector<float> output(std::size_t(tile_rows) * tile_columns + 8, sentinel);
    for (int row = 0; row < rows; ++row) for (int k = 0; k < inner; ++k) {
        const int pattern = ((row * 19 + k * 37) % 256) - 128;
        if (type == 8) {
            auto* block = weights.data() + std::size_t(row) * row_bytes + (k / 32) * 34;
            const auto scale = std::uint16_t(0x2000 + ((row + k / 32) % 5) * 0x180);
            std::memcpy(block, &scale, 2);
            block[2 + k % 32] = static_cast<unsigned char>(static_cast<std::int8_t>(pattern));
        } else {
            const auto half = std::uint16_t((pattern < 0 ? 0x8000 : 0) | 0x3400 | ((std::abs(pattern) * 7) & 1023));
            std::memcpy(weights.data() + std::size_t(row) * row_bytes + k * 2, &half, 2);
        }
    }
    const auto payload = TSGgml_ResidentWeightPayloadBytes(0, type, inner, rows, columns);
    require(payload > 0, "complete resident workspace query");
    void* session = nullptr;
    require(TSGgml_ResidentWeightCreate(0, type, inner, rows, columns, payload - 1, &session) == 0 && !session,
        "complete resident workspace cap is enforced before allocation");
    require(TSGgml_ResidentWeightCreate(0, type, inner, rows, columns, payload, &session) == 1,
        "create complete resident session");
    require(TSGgml_ResidentWeightProject(session) == 0, "complete projection rejects missing payload");
    require(TSGgml_ResidentWeightUploadRows(session, weights.data(), 1, 1) == 0, "weight coverage rejects a gap");
    require(TSGgml_WeightStreamingExecute(session, weights.data(), 1, output.data()) == 0,
        "tile API cannot reinterpret a complete session");
    for (int first = 0; first < rows; first += tile_rows) {
        const int count = std::min(tile_rows, rows - first);
        std::memcpy(staging.data(), weights.data() + std::size_t(first) * row_bytes, std::size_t(count) * row_bytes);
        require(TSGgml_ResidentWeightUploadRows(session, staging.data(), first, count) == 1, "consecutive complete weight rows");
        std::fill(staging.begin(), staging.end(), 0xA5);
    }
    require(TSGgml_ResidentWeightUploadRows(session, weights.data(), 0, 1) == 0, "loaded weights cannot be replaced");
    for (int revision = 0; revision < 2; ++revision) {
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = float(int((i * 7919 + 43 + revision * 331) % 65537) - 32768) / 32768.0f + float(i % 13) * 0x1p-22f;
        TensorView2DDesc source{input.data(), columns, inner, inner, 1, std::int64_t(input.size() * sizeof(float))};
        TensorView2DDesc result{reference.data(), columns, rows, rows, 1, std::int64_t(reference.size() * sizeof(float))};
        require(TSGgml_AddmmQuantF32(result, source, weights.data(), type, inner, rows, weights.size()) == 1,
            "unchanged complete resident reference");
        TSGgml_ClearHostBufferCache();
        require(TSGgml_ResidentWeightUploadInput(session, input.data(), 1, 1) == 0, "input coverage rejects a gap");
        for (int first = 0; first < columns; first += tile_columns) {
            const int count = std::min(tile_columns, columns - first);
            std::copy_n(input.data() + std::size_t(first) * inner, std::size_t(count) * inner, uploaded.data());
            require(TSGgml_ResidentWeightUploadInput(session, uploaded.data(), first, count) == 1, "consecutive complete input tokens");
            std::fill(uploaded.begin(), uploaded.end(), std::numeric_limits<float>::quiet_NaN());
            require(TSGgml_ResidentWeightDownload(session, output.data(), 0, 1, 0, 1) == 0,
                "replacement input invalidates old output before projection");
            if (first + count < columns)
                require(TSGgml_ResidentWeightProject(session) == 0, "partial input cannot project");
        }
        require(TSGgml_ResidentWeightProject(session) == 1 && TSGgml_ResidentWeightProject(session) == 1,
            "project original shape exactly once and reuse output");
        require(TSGgml_ResidentWeightDownload(session, output.data(), 0, 1, rows, 1) == 0,
            "out-of-range download does not poison computed output");
        for (int first_column = 0; first_column < columns; first_column += tile_columns) {
            const int count_columns = std::min(tile_columns, columns - first_column);
            for (int first_row = 0; first_row < rows; first_row += tile_rows) {
                const int count_rows = std::min(tile_rows, rows - first_row);
                std::fill(output.begin(), output.end(), sentinel);
                require(TSGgml_ResidentWeightDownload(session, output.data(), first_column, count_columns, first_row, count_rows) == 1,
                    "download complete output as bounded rectangles");
                require(std::all_of(output.begin() + std::size_t(count_rows) * count_columns, output.end(),
                    [sentinel](float value) { return value == sentinel; }), "complete output canary");
                for (int column = 0; column < count_columns; ++column)
                    std::copy_n(output.data() + std::size_t(column) * count_rows, count_rows,
                        actual.data() + std::size_t(first_column + column) * rows + first_row);
            }
        }
        require(std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); }), "complete projection finite");
        require(std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float)) == 0,
            "complete original shape must be bitwise equal to unchanged resident projection");
    }
    require(TSGgml_WeightStreamingDestroy(session) == 1, "release complete session");
    std::printf("resident-full type=%d K=%d M=%d N=%d transfer=%dx%d payload=%lld: 2 inputs byte-exact, complete coverage and canaries passed\n",
        type, inner, rows, columns, tile_rows, tile_columns, static_cast<long long>(payload));
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
        check_input_reuse(96, 63);
        check_f16(33, 129, 63, {33, 1, 9, 17, 33});
        TSGgml_Shutdown();
        std::puts("Q8 streaming singleton initialization passed.");
        return 0;
    }
    const int device = 0;
    require(TSGgml_MultiDeviceInit(3, &device, 1) == 1, "initialize CUDA");
    if (argc == 2 && std::strcmp(argv[1], "--resident-full") == 0) {
        check_resident_full(8, 96, 129, 9, 63, 4);
        check_resident_full(8, 2560, 3072, 36, 65, 7);
        check_resident_full(8, 2560, 6144, 65, 127, 32);
        check_resident_full(8, 2560, 20480, 36, 257, 32);
        check_resident_full(1, 2560, 128, 17, 63, 8);
        check_resident_full(1, 2560, 10752, 36, 257, 32);
        TSGgml_Shutdown();
        std::puts("Complete resident shape byte-exact checks passed against unchanged ggml.");
        return 0;
    }
    const bool resident_q8_only = argc == 2 && std::strcmp(argv[1], "--resident-q8") == 0;
    if (resident_q8_only || (argc == 2 && std::strcmp(argv[1], "--resident") == 0)) {
        for (int columns : {1, 8, 9, 16, 32, 36}) for (int tile : {1, 63, 128})
            check_resident(8, 2560, 256, columns, tile, std::min(columns, 32));
        // Exercise original-row fallback and a matrix-mode tail of one token.
        check_resident(8, 96, 129, 9, 63, 4);
        check_resident(8, 2560, 512, 36, 65, 7);
        if (!resident_q8_only) {
            for (int columns : {1, 8, 9, 16, 32, 36}) for (int tile : {1, 63, 128})
                check_resident(1, 2560, 256, columns, tile, std::min(columns, 32));
            check_resident(1, 2560, 10752, 36, 65, 32);
        }
        TSGgml_Shutdown();
        std::puts("Resident CUDA compatibility checks passed against unchanged ggml full Linear arithmetic.");
        return 0;
    }
    for (int inner : {32, 96, 1024}) for (int columns : {1, 8, 9, 17, 33})
        for (int tile : {1, 63, 64, 65}) check(inner, columns, tile);
    for (int inner : {32, 96, 1024}) for (int tile : {1, 63, 64, 65}) check_input_reuse(inner, tile);
    for (int inner : {1, 33, 65, 2560}) for (int tile : {1, 63, 64, 65})
        check_f16(inner, 129, tile, {33, 1, 17, 8, 9, 16, 32, 33});
    check_f16(2560, 10752, 65, {1, 9}); // Actual Gemma E4B per-layer projection shape.
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
    std::puts("Weight streaming: Q8 legacy ABI/reuse, F16 tails/model shape/reuse, ownership failure/retry, and no-cache checks passed.");
    return 0;
}
