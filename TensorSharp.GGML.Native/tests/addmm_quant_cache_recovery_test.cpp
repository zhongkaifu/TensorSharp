// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A failed cold bind must not leave an uninitialized DeviceCopy that a later
// AddmmQuant accepts as filled. This injects bounded return/exception failures
// after binding, not an actual OOM. CUDA must observe the pending device copy;
// CPU is a separate failure-status/numerical-recovery control, not GPU proof.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
struct TensorView2DDesc {
    void* data;
    int dim0, dim1, stride0, stride1;
    std::int64_t raw_bytes;
};
int TSGgml_IsBackendAvailable(int);
const char* TSGgml_GetLastError();
int TSGgml_AddmmQuantF32(TensorView2DDesc, TensorView2DDesc, void*, int,
    std::int64_t, std::int64_t, std::int64_t, int);
void TSGgml_MatmulTestAbortAfterWeightBind(int);
int TSGgml_MatmulTestAbortedPendingDeviceCopy();
int TSGgml_HasResidentDeviceCopy(void*, std::int64_t);
std::int64_t TSGgml_DeviceCopyCacheResidentBytes();
int TSGgml_PreloadQuantizedWeight(void*, void*, int, std::int64_t, std::int64_t, std::int64_t);
void TSGgml_ClearHostBufferCache();
void TSGgml_InvalidateHostBuffer(void*);
void TSGgml_ReleaseReuseComputeBuffers();
void TSGgml_Shutdown();
}

namespace {
constexpr int width = 64;
constexpr float sentinel = -901.25f;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Weight {
    int type;
    std::vector<std::uint8_t> bytes;
    std::vector<float> represented;

    explicit Weight(bool quantized) : type(quantized ? 8 : 0), represented(width * width) {
        // Q8_0: 32 signed int8 values with an exactly represented F16 1/32 scale.
        // The row length and every row's offset respect the quant block size.
        bytes.resize(quantized ? width * (width / 32) * 34 : represented.size() * sizeof(float));
        for (int row = 0; row < width; ++row) {
            for (int col = 0; col < width; ++col) {
                const auto q = static_cast<std::int8_t>((row * 17 + col * 7) % 29 - 14);
                represented[row * width + col] = float(q) / 32.f;
                if (quantized) {
                    const std::size_t block = (row * (width / 32) + col / 32) * 34;
                    const std::uint16_t scale = 0x2800;
                    std::memcpy(bytes.data() + block, &scale, sizeof(scale));
                    std::memcpy(bytes.data() + block + 2 + col % 32, &q, sizeof(q));
                }
            }
        }
        if (!quantized) std::memcpy(bytes.data(), represented.data(), bytes.size());
        require(bytes.size() >= 4096, "Fixture does not reach cacheable weight threshold");
    }
};

TensorView2DDesc view(std::vector<float>& data, int rows) {
    return {data.data(), rows, width, width, 1, std::int64_t(data.size() * sizeof(float))};
}

void fill_input(std::vector<float>& input, int rows, int generation) {
    std::fill(input.begin(), input.end(), 0.f);
    for (int row = 0; row < rows; ++row) {
        // Equal-magnitude sparse inputs use 127 * a power-of-two scale, exactly
        // represented by both CPU and CUDA Q8 activation quantizers (including
        // CPU's F16 scale). This avoids a dense-activation rounding confound.
        const int column = (row * 19 + generation * 11) % width;
        input[row * width + column] = row % 2 == 0 ? 127.f / 128.f : -127.f / 256.f;
        input[row * width + (column + 31) % width] = -input[row * width + column];
    }
}

void check_scalar(const Weight& weight, const std::vector<float>& input,
                  const std::vector<float>& output, int rows) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < width; ++col) {
            double expected = 0;
            for (int inner = 0; inner < width; ++inner)
                expected += double(input[row * width + inner]) * weight.represented[col * width + inner];
            const float actual = output[row * width + col];
            if (!std::isfinite(actual) || std::abs(double(actual) - expected) > 1e-5) {
                std::fprintf(stderr, "Scalar mismatch type=%d row=%d col=%d expected=%.12g actual=%.12g\n",
                    weight.type, row, col, expected, double(actual));
                throw std::runtime_error("AddmmQuant recovery differs from represented-weight scalar oracle");
            }
        }
    }
}

void check_case(bool cuda, bool quantized, bool opaque, int rows) {
    TSGgml_ClearHostBufferCache();
    TSGgml_ReleaseReuseComputeBuffers();
    Weight weight(quantized);
    int opaque_key = 0;
    void* key = opaque ? static_cast<void*>(&opaque_key) : weight.bytes.data();
    const auto byte_count = std::int64_t(weight.bytes.size());
    std::vector<float> input(rows * width), output(rows * width, sentinel);
    fill_input(input, rows, 0);

    if (opaque) {
        require(cuda, "Opaque-key case is a GPU preload/reload contract");
        require(TSGgml_PreloadQuantizedWeight(key, weight.bytes.data(), weight.type,
            width, width, byte_count) == 1, "Opaque-key preload failed");
        require(TSGgml_HasResidentDeviceCopy(key, byte_count) == 1, "Preload was not resident");
        TSGgml_InvalidateHostBuffer(key); // Registration deliberately survives eviction.
    }
    auto run = [&]() {
        return TSGgml_AddmmQuantF32(view(output, rows), view(input, rows),
            key, weight.type, width, width, byte_count, 0);
    };
    require(run() == 1, "Initial AddmmQuant control failed");
    check_scalar(weight, input, output, rows);
    const auto baseline = output;

    for (int mode : {1, 2}) {
        TSGgml_InvalidateHostBuffer(key);
        require(TSGgml_HasResidentDeviceCopy(key, byte_count) == 0, "Cold precondition failed");
        const auto empty_resident_bytes = TSGgml_DeviceCopyCacheResidentBytes();
        std::fill(output.begin(), output.end(), sentinel);
        TSGgml_MatmulTestAbortAfterWeightBind(mode);
        require(run() == 0, "Injected cold failure was ignored");
        require(TSGgml_MatmulTestAbortedPendingDeviceCopy() == int(cuda),
            "Failure did not exercise the expected pending DeviceCopy state");
        require(TSGgml_GetLastError() != nullptr && TSGgml_GetLastError()[0] != '\0',
            "Failure lost its diagnostic");
        require(TSGgml_HasResidentDeviceCopy(key, byte_count) == 0,
            "Abandoned cold bind left an uninitialized resident entry");
        require(TSGgml_DeviceCopyCacheResidentBytes() == empty_resident_bytes,
            "Abandoned cold bind leaked resident-byte accounting");
        require(std::all_of(output.begin(), output.end(), [](float x) { return x == sentinel; }),
            "Failed call modified output before graph execution");

        // The hook is one-shot; recovery must work without resetting the backend
        // or clearing other caches and must reproduce the successful baseline.
        require(run() == 1, "Cold-failure retry failed");
        require(output == baseline, "Cold-failure retry changed output");
        check_scalar(weight, input, output, rows);
        require(TSGgml_HasResidentDeviceCopy(key, byte_count) == int(cuda),
            "Successful retry did not establish expected residency");

        const auto warm_resident_bytes = TSGgml_DeviceCopyCacheResidentBytes();
        std::fill(output.begin(), output.end(), sentinel);
        TSGgml_MatmulTestAbortAfterWeightBind(mode);
        require(run() == 0, "Injected warm failure was ignored");
        require(TSGgml_MatmulTestAbortedPendingDeviceCopy() == 0,
            "Existing initialized entry was classified as a new DeviceCopy");
        require(TSGgml_HasResidentDeviceCopy(key, byte_count) == int(cuda) &&
            TSGgml_DeviceCopyCacheResidentBytes() == warm_resident_bytes,
            "Warm failure invalidated an already initialized entry");
        require(std::all_of(output.begin(), output.end(), [](float x) { return x == sentinel; }),
            "Warm failure modified output");
        require(run() == 1 && output == baseline, "Warm-failure retry changed output");
    }

    fill_input(input, rows, 1);
    require(run() == 1, "Changed-input warm call failed");
    check_scalar(weight, input, output, rows);
    require(output != baseline, "Fixture failed to distinguish changed input");
    const auto changed = output;
    TSGgml_InvalidateHostBuffer(key);
    require(run() == 1 && output == changed, "Explicit eviction/reload changed output");
    check_scalar(weight, input, output, rows);
    TSGgml_ClearHostBufferCache();
    require(TSGgml_HasResidentDeviceCopy(key, byte_count) == 0, "Final cache clear failed");
    std::printf("PASS type=%s key=%s rows=%d cold-return/cold-throw/warm-preservation/reload%s\n",
        quantized ? "Q8_0" : "F32", opaque ? "opaque" : "host", rows,
        cuda ? " actual-DeviceCopy" : " CPU-control-only");
}
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "cpu") != 0 && std::strcmp(argv[1], "cuda") != 0)) return 2;
    const bool cuda = std::strcmp(argv[1], "cuda") == 0;
    if (!TSGgml_IsBackendAvailable(cuda ? 3 : 2)) return 77;
    try {
        for (bool quantized : {false, true})
            for (int rows : {1, 3})
                for (bool opaque : {false, true})
                    if (!opaque || cuda) check_case(cuda, quantized, opaque, rows);
        TSGgml_ReleaseReuseComputeBuffers();
        TSGgml_Shutdown();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s; native: %s\n", error.what(),
            TSGgml_GetLastError() ? TSGgml_GetLastError() : "(none)");
        TSGgml_ClearHostBufferCache();
        TSGgml_ReleaseReuseComputeBuffers();
        TSGgml_Shutdown();
        return 1;
    }
}
