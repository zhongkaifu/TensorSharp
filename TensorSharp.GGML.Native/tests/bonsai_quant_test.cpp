// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "bonsai_quant.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

// Independent encoder: logical ternary values -> ordinary base-3 integer ->
// published fixed-point byte representation. Expected values never use the
// production unpacker/transcoder or upstream's dequantizer.
void encode(int type, const std::array<int, 128> & codes, ggml_fp16_t scale, uint8_t * dst) {
    std::memset(dst, 0, tsg_bonsai_row_size(type, 128));
    if (type == TSG_BONSAI_PQ2_0) {
        std::memcpy(dst, &scale, 2);
        for (int j = 0; j < 128; ++j) dst[2 + j / 4] |= codes[j] << (2 * (j % 4));
        return;
    }
    const auto encode_lanes = [&](int src_base, int dst_base, int lanes, int digits) {
        for (int lane = 0; lane < lanes; ++lane) {
            unsigned integer = 0;
            for (int digit = 0; digit < digits; ++digit)
                integer = 3 * integer + codes[src_base + lane + digit * lanes];
            if (digits == 4) integer *= 3;
            dst[dst_base + lane] = (256 * integer + 242) / 243;
        }
    };
    encode_lanes(0, 0, 16, 5);
    encode_lanes(80, 16, 8, 5);
    encode_lanes(120, 24, 2, 4);
    std::memcpy(dst + 26, &scale, 2);
}

struct fixture {
    std::vector<uint8_t> packed, q2;
    std::vector<float> expected;
};

fixture make_fixture(int type, int blocks, bool scale_edges = false) {
    fixture f;
    const size_t stride = tsg_bonsai_row_size(type, 128);
    f.packed.resize(size_t(blocks) * stride);
    f.q2.resize(size_t(blocks) * 36);
    f.expected.resize(size_t(blocks) * 128);
    const ggml_fp16_t edges[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0x3c00, 0xbc00, 0x7bff};
    for (int block = 0; block < blocks; ++block) {
        const ggml_fp16_t bits = scale_edges ? edges[block % (sizeof(edges) / sizeof(edges[0]))]
            : ggml_fp32_to_fp16(float(1 + block % 7) / 64.0f);
        const float d = ggml_fp16_to_fp32(bits);
        std::array<int, 128> codes{};
        // Across 243 consecutive blocks, every five-trit tuple occurs in
        // each payload lane, including the 16/8-stage and tail boundaries.
        for (int j = 0; j < 128; ++j) {
            if (type == TSG_BONSAI_PQ2_0) codes[j] = (block + j * 7 + j / 4) % 4;
            else {
                const int base = j < 80 ? 0 : j < 120 ? 80 : 120;
                const int lanes = j < 80 ? 16 : j < 120 ? 8 : 2;
                int value = (block + (j - base) % lanes * 17) % 243;
                for (int digit = 0; digit < (j - base) / lanes; ++digit) value /= 3;
                codes[j] = value % 3;
            }
            f.expected[size_t(block) * 128 + j] = float(codes[j] - 1) * d;
        }
        encode(type, codes, bits, f.packed.data() + size_t(block) * stride);
    }
    require(tsg_bonsai_transcode_q2_0(type, f.packed.data(), int64_t(blocks) * 128, f.q2.data()) == 0,
            "lossless transcode failed");
    return f;
}

void codec_tests() {
    require(tsg_bonsai_row_size(142, 128) == 34 && tsg_bonsai_row_size(143, 256) == 56, "row sizes");
    for (int type : {142, 143}) {
        require(tsg_bonsai_row_size(type, -128) == 0 && tsg_bonsai_row_size(type, 0) == 0 &&
                tsg_bonsai_row_size(type, 127) == 0, "invalid row sizes");
        const auto f = make_fixture(type, 2430, true);
        std::vector<float> decoded(f.expected.size()), upstream(f.expected.size());
        require(tsg_bonsai_dequantize(type, f.packed.data(), int64_t(decoded.size()), decoded.data()) == 0,
                "private-type dequantization failed");
        ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(f.q2.data(), upstream.data(), int64_t(upstream.size()));
        require(std::memcmp(decoded.data(), f.expected.data(), decoded.size() * sizeof(float)) == 0,
                "private dequantization changed a scale or ternary code");
        require(std::memcmp(upstream.data(), f.expected.data(), upstream.size() * sizeof(float)) == 0,
                "upstream Q2_0 does not exactly match private format");
        std::array<uint8_t, 80> guard; guard.fill(0xa5);
        const auto before = guard;
        require(tsg_bonsai_transcode_q2_0(type, f.packed.data(), 127, guard.data()) == -1, "unaligned transcode accepted");
        require(tsg_bonsai_transcode_q2_0(type, nullptr, 128, guard.data()) == -1, "null transcode accepted");
        require(tsg_bonsai_transcode_q2_0(type, guard.data(), 128, guard.data() + 1) == -1, "overlap accepted");
        require(tsg_bonsai_transcode_q2_0(type, guard.data() + 1, 128, guard.data()) == -1, "reverse overlap accepted");
        require(guard == before, "invalid input wrote destination");
    }
    require(tsg_bonsai_row_size(42, 128) == 0, "private decoder claimed upstream type");
}

void matmul_test(ggml_backend_t backend, int type, int inner, int rows, int columns) {
    auto f = make_fixture(type, inner * rows / 128);
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "context allocation");
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_0, inner, rows);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner, columns);
    auto * output = ggml_mul_mat(ctx, weights, input);
    ggml_prec_set_acc(output, GGML_PREC_F32);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    require(ggml_backend_supports_op(backend, output), "backend does not support upstream Q2_0 matmul");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "backend allocation");
    std::vector<float> x(size_t(inner) * columns), actual(size_t(rows) * columns);
    for (int col = 0; col < columns; ++col)
        for (int k = 0; k < inner; ++k)
            // Integer/64 activations with amax=127/64 in each 32-element
            // group are exactly representable in CPU's Q8 activation path.
            x[size_t(col) * inner + k] = float(k % 32 == 0 ? 127 : (k * 37 + col * 11) % 255 - 127) / 64.0f;
    ggml_backend_tensor_set(weights, f.q2.data(), 0, f.q2.size());
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Q2 matmul compute");
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    double max_error = 0;
    for (int col = 0; col < columns; ++col)
        for (int row = 0; row < rows; ++row) {
            double expected = 0;
            for (int k = 0; k < inner; ++k)
                expected += double(f.expected[size_t(row) * inner + k]) * x[size_t(col) * inner + k];
            const double error = std::abs(double(actual[size_t(col) * rows + row]) - expected);
            require(std::isfinite(error), "nonfinite matmul result");
            max_error = std::max(max_error, error);
        }
    std::printf("%s type=%d [%d,%d] x %d max_abs_error=%.9g\n", ggml_backend_name(backend), type, inner, rows, columns, max_error);
    require(max_error <= 2e-3, "Q2 matmul disagrees with independent dense oracle");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}
} // namespace

int main(int argc, char ** argv) {
    const char * name = argc > 1 ? argv[1] : "cpu";
    codec_tests();
    ggml_backend_t backend = nullptr;
    if (std::strcmp(name, "cpu") == 0) { backend = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend, 4); }
#ifdef TSG_GGML_USE_METAL
    if (std::strcmp(name, "metal") == 0) backend = ggml_backend_metal_init();
#endif
#ifdef TSG_GGML_USE_CUDA
    if (std::strcmp(name, "cuda") == 0) backend = ggml_backend_cuda_init(0);
#endif
    if (!backend) { std::fprintf(stderr, "SKIP: %s backend unavailable\n", name); return 77; }
    for (int type : {142, 143})
        for (int inner : {128, 384, 5120})
            for (int columns : {1, 4, 32}) matmul_test(backend, type, inner, 65, columns);
    ggml_backend_free(backend);
    std::puts("PASS: exhaustive ternary tuples, scale edge cases, validation guards and native Q2_0 projections");
    return 0;
}
