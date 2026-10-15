// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_matmul_precision.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml_ops_dsv4_fused.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include "precision_test_utils.h"

namespace {
constexpr int block_elements = 256;
constexpr size_t block_bytes = 176;
constexpr size_t scale_offset = 4, high_offset = 16, low_offset = 48;
constexpr double maximum_error = 5e-5, relative_error = 5e-6;

// The fixture treats unchanged upstream ggml's Q5_K declaration as a wire-format
// specification only. No production quantizer, dequantizer, half conversion,
// block structure or dot implementation participates in the FP64 dot oracle.
// Unchanged upstream's decoder is used separately to check format compatibility.
void store_half(unsigned char* destination, uint16_t bits) {
    destination[0] = static_cast<unsigned char>(bits);
    destination[1] = static_cast<unsigned char>(bits >> 8);
}

double half_bits(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    require(exponent != 31, "Q5 fixture contains a nonfinite half scale");
    const double magnitude = exponent == 0 ? std::ldexp(double(fraction), -24)
        : std::ldexp(double(1024 + fraction), int(exponent) - 25);
    return bits & 0x8000 ? -magnitude : magnitude;
}

double stored_half(const unsigned char* source) {
    return half_bits(unsigned(source[0]) | (unsigned(source[1]) << 8));
}

void pack_group_fields(unsigned char* destination, const std::array<unsigned, 8>& scales,
                       const std::array<unsigned, 8>& minima) {
    std::memset(destination, 0, 12);
    // Pack individual bits, independently of the whole-field expressions in
    // the decoder below. Groups 4..7 share high bits with groups 0..3.
    for (unsigned group = 0; group < 8; ++group) {
        require(scales[group] <= 63 && minima[group] <= 63, "Q5 group field exceeds six bits");
        for (unsigned bit = 0; bit < 6; ++bit) {
            const unsigned scale_byte = group < 4 ? group : bit < 4 ? group + 4 : group - 4;
            const unsigned scale_bit = group < 4 || bit < 4 ? bit : bit + 2;
            const unsigned minimum_byte = group < 4 ? group + 4 : bit < 4 ? group + 4 : group;
            const unsigned minimum_bit = group < 4 ? bit : bit < 4 ? bit + 4 : bit + 2;
            destination[scale_byte] |= ((scales[group] >> bit) & 1) << scale_bit;
            destination[minimum_byte] |= ((minima[group] >> bit) & 1) << minimum_bit;
        }
    }
}

void pack_quant(unsigned char* block, unsigned element, unsigned quant) {
    require(element < 256 && quant < 32, "Q5 fixture quant code is out of range");
    const unsigned group = element / 32, lane = element % 32;
    auto& low = block[low_offset + (group / 2) * 32 + lane];
    const unsigned shift = (group % 2) * 4;
    low = static_cast<unsigned char>((low & ~(15u << shift)) | ((quant & 15) << shift));
    auto& high = block[high_offset + lane];
    high = static_cast<unsigned char>((high & ~(1u << group)) | ((quant >> 4) << group));
}

unsigned stored_quant(const unsigned char* block, unsigned element) {
    const unsigned group = element / 32, lane = element % 32;
    const unsigned low = (block[low_offset + (element / 64) * 32 + lane] >> (group % 2 ? 4 : 0)) & 15;
    return low | (((block[high_offset + lane] >> group) & 1) << 4);
}

double stored_weight(const unsigned char* block, unsigned element) {
    const unsigned group = element / 32;
    const auto* fields = block + scale_offset;
    const unsigned scale = group < 4 ? fields[group] & 63
        : (fields[group + 4] & 15) | ((fields[group - 4] >> 6) << 4);
    const unsigned minimum = group < 4 ? fields[group + 4] & 63
        : (fields[group + 4] >> 4) | ((fields[group] >> 6) << 4);
    return stored_half(block) * scale * stored_quant(block, element)
        - stored_half(block + 2) * minimum;
}

void check_format_oracle() {
    const uint16_t endian_probe = 1;
    require(*reinterpret_cast<const unsigned char*>(&endian_probe) == 1,
        "Q5 checkpoint byte fixture requires a little-endian host");
    require(ggml_blck_size(GGML_TYPE_Q5_K) == block_elements
        && ggml_type_size(GGML_TYPE_Q5_K) == block_bytes,
        "Q5 fixture requires the declared 256-value/176-byte checkpoint format");
    std::array<unsigned char, block_bytes> block{};
    const std::array<unsigned, 8> scales{0, 1, 31, 32, 33, 47, 62, 63};
    const std::array<unsigned, 8> minima{63, 62, 47, 33, 32, 31, 1, 0};
    const std::array<unsigned char, 12> expected_fields{128, 129, 223, 224, 191, 126, 47, 33, 1, 255, 30, 15};
    pack_group_fields(block.data() + scale_offset, scales, minima);
    require(std::memcmp(block.data() + scale_offset, expected_fields.data(), expected_fields.size()) == 0,
        "Independent Q5 bit packer disagrees with the declared six-bit layout");
    store_half(block.data(), 0x3c00);   // +1
    store_half(block.data() + 2, 0x3800); // +1/2
    for (unsigned element = 0; element < 256; ++element)
        pack_quant(block.data(), element, (element / 32 * 4 + element % 32) % 32);
    require(block[high_offset] == 0xf0 && block[low_offset] == 0x40
        && block[low_offset + 32] == 0xc8 && block[low_offset + 64] == 0x40
        && block[low_offset + 96] == 0xc8,
        "Independent Q5 bit packer disagrees with the declared fifth-bit layout");
    for (unsigned element = 0; element < 256; ++element) {
        const unsigned group = element / 32, quant = (group * 4 + element % 32) % 32;
        require(stored_quant(block.data(), element) == quant
            && stored_weight(block.data(), element) == double(scales[group] * quant) - double(minima[group]) / 2,
            "Independent Q5 stored-byte oracle failed a declared format example");
    }
    store_half(block.data(), 0x0001); // smallest positive subnormal
    store_half(block.data() + 2, 0x8001);
    require(stored_half(block.data()) == std::ldexp(1.0, -24)
        && stored_half(block.data() + 2) == -std::ldexp(1.0, -24),
        "Independent half oracle failed its signed subnormal example");
    store_half(block.data(), 0x8000);
    require(stored_half(block.data()) == 0 && std::signbit(stored_half(block.data())),
        "Independent half oracle failed its signed zero example");
    std::printf("SENSITIVE_Q5_FORMAT block_values=256 block_bytes=176 scale_groups=8 scale_bits=6 quant_bits=5 independent_format_examples=1\n");
}

struct q5_checkpoint {
    ggml_tensor* storage;
    ggml_tensor* tensor;
    int inner, rows;
    size_t row_bytes;
    std::vector<unsigned char> bytes;
    std::vector<double> oracle;

    q5_checkpoint(ggml_context* ctx, int k, int m, bool padded, bool first_stage,
                  bool narrow_cancellation = false)
        : inner(k), rows(m), row_bytes(size_t(k / block_elements + (padded ? 1 : 0)) * block_bytes) {
        require(k > 0 && k % block_elements == 0, "Q5 fixture inner dimension must contain full blocks");
        storage = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, k + (padded ? block_elements : 0), m + 2);
        tensor = ggml_view_2d(ctx, storage, k, m, row_bytes, row_bytes);
        bytes.resize(row_bytes * size_t(m + 2), 0xa5);
        oracle.resize(size_t(k) * m);
        for (int row = 0; row < m; ++row) {
            for (int index = 0; index < k / block_elements; ++index) {
                auto* block = bytes.data() + row_bytes * size_t(row + 1) + size_t(index) * block_bytes;
                std::memset(block, 0, block_bytes);
                if (narrow_cancellation) {
                    // Exact +/-1 checkpoint weights, with opposite signs for
                    // adjacent K elements and alternating phase between rows.
                    // Retain the declared golden values before byte packing.
                    store_half(block, 0x3c00); store_half(block + 2, 0x3c00);
                    std::array<unsigned, 8> scales{}, minima{};
                    scales.fill(1); minima.fill(1);
                    for (unsigned element = 0; element < block_elements; ++element) {
                        const unsigned quant = (unsigned(row) + element) % 2 == 0 ? 2 : 0;
                        oracle[size_t(row) * k + size_t(index) * block_elements + element]
                            = double(quant) - 1.0;
                        pack_quant(block, element, quant);
                    }
                    pack_group_fields(block + scale_offset, scales, minima);
                    continue;
                }
                // Modest binary scales keep the declared F32 error bounds useful.
                // Include signed and zero super-scales and a subnormal scale.
                uint16_t d = uint16_t(0x0400 + (row * 37 + index * 59) % 1024);
                uint16_t minimum = uint16_t(0x1000 + (row * 19 + index * 23) % 1024);
                if (row % 5 == 0) d |= 0x8000;
                if (row % 7 == 0) minimum |= 0x8000;
                if (row % 17 == 0) d = 0;
                if (row % 19 == 0) minimum = 0;
                if (row == 1 && index == 0) d = 0x0201;
                store_half(block, d); store_half(block + 2, minimum);
                std::array<unsigned, 8> scales{}, minima{};
                for (unsigned group = 0; group < 8; ++group) {
                    scales[group] = (unsigned(row) * 17 + unsigned(index) * 5 + group * 13) % 64;
                    minima[group] = (unsigned(row) * 29 + unsigned(index) * 11 + group * 7 + 3) % 64;
                    for (unsigned lane = 0; lane < 32; ++lane) {
                        const unsigned quant = (unsigned(row) * 13 + unsigned(index) * 19 + group * 7 + lane * 11) % 32;
                        // Keep the golden value BEFORE packing any format fields.
                        // Neither the owned nor upstream decoder defines this oracle.
                        oracle[size_t(row) * k + size_t(index) * block_elements + group * 32 + lane]
                            = half_bits(d) * scales[group] * quant - half_bits(minimum) * minima[group];
                        pack_quant(block, group * 32 + lane, quant);
                    }
                }
                pack_group_fields(block + scale_offset, scales, minima);
            }
        }
        if (!narrow_cancellation) {
            // Exact first row: stage one computes x[0]-x[1], stage two selects
            // that residual. Every remaining weight in these rows is exactly zero.
            for (int index = 0; index < k / block_elements; ++index)
                std::memset(bytes.data() + row_bytes + size_t(index) * block_bytes, 0, block_bytes);
            auto* residual = bytes.data() + row_bytes;
            store_half(residual, 0x3c00);
            std::array<unsigned, 8> scales{}, minima{};
            scales[0] = 1;
            if (first_stage) {
                store_half(residual + 2, 0x3c00); minima[0] = 1;
                for (unsigned lane = 0; lane < 32; ++lane) pack_quant(residual, lane, 1);
                pack_quant(residual, 0, 2); pack_quant(residual, 1, 0);
            } else pack_quant(residual, 0, 1);
            pack_group_fields(residual + scale_offset, scales, minima);
            std::fill(oracle.begin(), oracle.begin() + k, 0.0);
            oracle[0] = 1.0;
            if (first_stage) oracle[1] = -1.0;
        }
        // ggml_init has already initialized the half-conversion tables before
        // this constructor. Decode each complete original-format row, excluding
        // padding, solely as a wire-format compatibility check. Dot products
        // below continue to use the retained prepack FP64 golden values.
        const auto* traits = ggml_get_type_traits(GGML_TYPE_Q5_K);
        require(traits && traits->to_float, "Unchanged upstream Q5 format decoder is unavailable");
        std::vector<float> upstream(size_t(k), 0.0f);
        bool positive = false, negative = false, zero = false;
        for (int row = 0; row < m; ++row) {
            traits->to_float(bytes.data() + row_bytes * size_t(row + 1), upstream.data(), k);
            for (int element = 0; element < k; ++element) {
                const auto* block = bytes.data() + row_bytes * size_t(row + 1)
                    + size_t(element / block_elements) * block_bytes;
                const double value = oracle[size_t(row) * k + element];
                require(stored_weight(block, unsigned(element % block_elements)) == value,
                    "Independent Q5 stored-byte decoder disagrees with prepack FP64 golden values");
                const float rounded = float(value);
                require(std::memcmp(&upstream[size_t(element)], &rounded, sizeof(float)) == 0,
                    "Q5 packed format disagrees with unchanged upstream decoding of prepack golden values");
                positive |= value > 0; negative |= value < 0; zero |= value == 0;
            }
        }
        require(positive && negative && (narrow_cancellation || zero),
            "Q5 fixture must contain positive/negative weights and chain fixtures must also contain zero weights");
        std::printf("SENSITIVE_Q5_FORMAT_COMPAT inner=%d rows=%d checked_values=%zu prepack_fp64=1 independent_stored_decoder=1 unchanged_upstream_to_float=1 oracle_from_upstream=0 narrow_cancellation=%d\n",
            k, m, oracle.size(), int(narrow_cancellation));
    }

    void upload() const { ggml_backend_tensor_set(storage, bytes.data(), 0, bytes.size()); }
    void check_unchanged() const {
        std::vector<unsigned char> actual(bytes.size());
        ggml_backend_tensor_get(storage, actual.data(), 0, actual.size());
        require(actual == bytes, "Sensitive Q5 projection modified checkpoint bytes, row padding or guards");
    }
};

struct chain_result { std::vector<float> first, second; };

std::pair<double, double> measure(const std::vector<float>& values, const std::vector<double>& oracle) {
    double maximum = 0, squared = 0, reference_squared = 0;
    for (size_t index = 0; index < values.size(); ++index) {
        require(std::isfinite(values[index]), "Q5 projection emitted a nonfinite value");
        const double delta = double(values[index]) - oracle[index];
        maximum = std::max(maximum, std::abs(delta));
        squared += delta * delta; reference_squared += oracle[index] * oracle[index];
    }
    require(reference_squared > 0, "Q5 reference must have nonzero norm");
    const double relative = std::sqrt(squared / reference_squared);
    require(maximum <= maximum_error && relative <= relative_error,
        "Q5 sensitive chain failed its predeclared FP64 error gates");
    return {maximum, relative};
}

chain_result check_chain(ggml_backend_t backend, int tokens, int inner = 512,
                         bool padded_weights = true, bool interleaved_input = false, int rows = 128) {
    constexpr int rank = 256;
    auto* ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot create Q5 sensitive projection context");
    q5_checkpoint a(ctx, inner, rank, padded_weights, true), b(ctx, rank, rows, padded_weights, false);
    input_tensor x(ctx, GGML_TYPE_F32, {inner, tokens, 1, 1}, true, interleaved_input);
    const auto role = tsg_projection_role::recurrent_decay;
    require(tsg_projection_weight(ctx, a.tensor, role) == a.tensor
        && tsg_projection_weight(ctx, b.tensor, tsg_projection_role::residual_mixing) == b.tensor,
        "Sensitive Q5 projection widened resident checkpoint weights");
    require(tsg_projection_quant_f32_if_required(ctx, a.tensor, x.tensor, tsg_projection_role::throughput) == nullptr,
        "Ordinary throughput Q5 projection changed construction");
    for (auto sensitive_role : {tsg_projection_role::recurrent_decay, tsg_projection_role::recurrent_output_gate,
            tsg_projection_role::recurrent_update, tsg_projection_role::residual_mixing, tsg_projection_role::routing_logits}) {
        auto* node = tsg_projection_quant_f32_if_required(ctx, a.tensor, x.tensor, sensitive_role);
        require(node && tsg_is_matmul_quant_f32(node) && !tsg_is_matmul_q8_f32(node)
            && node->src[0] == a.tensor && node->src[1] == x.tensor,
            "A sensitive Q5 projection role bypassed the owned original-format operator");
    }
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_K}) {
        auto* other = ggml_new_tensor_2d(ctx, type, inner, 1);
        require(tsg_projection_quant_f32_if_required(ctx, other, x.tensor, role) == nullptr,
            "Q5 factory changed construction for an unsupported source format");
    }
    auto* first = tsg_projection_quant_f32_if_required(ctx, a.tensor, x.tensor, role);
    require(first != nullptr, "First sensitive Q5 role bypassed the owned F32-activation operator");
    auto* second = tsg_projection_quant_f32_if_required(ctx, b.tensor, first, tsg_projection_role::residual_mixing);
    require(first && second && tsg_is_matmul_quant_f32(first) && tsg_is_matmul_quant_f32(second),
        "Chained sensitive Q5 role bypassed the owned F32-activation operator");
    require(a.tensor->type == GGML_TYPE_Q5_K && b.tensor->type == GGML_TYPE_Q5_K
        && first->src[0] == a.tensor && first->src[1] == x.tensor
        && second->src[0] == b.tensor && second->src[1] == first
        && first->type == GGML_TYPE_F32 && second->type == GGML_TYPE_F32
        && first->ne[0] == rank && second->ne[0] == rows
        && first->ne[1] == tokens && second->ne[1] == tokens
        && first->ne[2] == 1 && first->ne[3] == 1 && second->ne[2] == 1 && second->ne[3] == 1
        && a.tensor->nb[0] == block_bytes && a.tensor->nb[1] == a.row_bytes
        && b.tensor->nb[0] == block_bytes && b.tensor->nb[1] == b.row_bytes
        && x.tensor->type == GGML_TYPE_F32 && x.tensor->nb[0] == sizeof(float) * (interleaved_input ? 2 : 1)
        && ggml_is_contiguous(first) && ggml_is_contiguous(second),
        "Sensitive Q5 projection changed source format, ownership, strides or output layout");
    ggml_set_output(first);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, second);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "Cannot allocate Q5 sensitive projection fixture");
    const void* a_pointer = a.tensor->data;
    const void* b_pointer = b.tensor->data;
    const void* x_pointer = x.tensor->data;
    require(a_pointer && b_pointer && x_pointer, "Q5 source tensors have no allocated resident storage");
    for (int column = 0; column < tokens; ++column) {
        for (int element = 0; element < inner; ++element)
            x.values[size_t(column) * inner + element] = std::sin(float(element * 13 + column * 7) * 0.019f) * 0.8f
                + float((element + column) % 17) * 0.000013f;
        x.values[size_t(column) * inner] = 1.0f + std::ldexp(float(column % 4 + 1), -16);
        x.values[size_t(column) * inner + 1] = 1.0f;
    }
    a.upload(); b.upload(); x.upload();
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
        "Q5 sensitive projection graph execution failed");
    ggml_backend_synchronize(backend);
    chain_result actual{std::vector<float>(size_t(rank) * tokens), std::vector<float>(size_t(rows) * tokens)};
    ggml_backend_tensor_get(first, actual.first.data(), 0, actual.first.size() * sizeof(float));
    ggml_backend_tensor_get(second, actual.second.data(), 0, actual.second.size() * sizeof(float));
    std::vector<double> oracle_first(actual.first.size()), oracle_second(actual.second.size());
    for (int column = 0; column < tokens; ++column) {
        for (int row = 0; row < rank; ++row) {
            double sum = 0;
            for (int element = 0; element < inner; ++element)
                sum += a.oracle[size_t(row) * inner + element] * x.at(element, column);
            oracle_first[size_t(column) * rank + row] = sum;
        }
        for (int row = 0; row < rows; ++row) {
            double sum = 0;
            for (int element = 0; element < rank; ++element)
                sum += b.oracle[size_t(row) * rank + element] * oracle_first[size_t(column) * rank + element];
            oracle_second[size_t(column) * rows + row] = sum;
        }
        const float residual = x.at(0, column) - x.at(1, column);
        require(residual > 0 && actual.first[size_t(column) * rank] == residual
            && actual.second[size_t(column) * rows] == residual,
            "Q5 projection quantized away the F32 activation residual");
    }
    const auto first_error = measure(actual.first, oracle_first), second_error = measure(actual.second, oracle_second);
    require(a.tensor->data == a_pointer && b.tensor->data == b_pointer && x.tensor->data == x_pointer
        && first->src[0] == a.tensor && first->src[1] == x.tensor && second->src[0] == b.tensor && second->src[1] == first,
        "Q5 projection changed resident input pointers or source ownership during execution");
    a.check_unchanged(); b.check_unchanged(); x.check_unchanged();
    std::printf("SENSITIVE_Q5_PROJECTION backend=%s inner=%d rank=%d rows=%d columns=%d full_values=%zu residual_columns=%d first_max_abs=%.9g first_relative_l2=%.9g second_max_abs=%.9g second_relative_l2=%.9g resident_q5=1 padded_weights=%d interleaved_input=%d\n",
        ggml_backend_name(backend), inner, rank, rows, tokens, actual.first.size() + actual.second.size(), tokens,
        first_error.first, first_error.second, second_error.first, second_error.second, int(padded_weights), int(interleaved_input));
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return actual;
}

float rounded_f32(double value) {
    // Force an F32 rounding boundary independently of host register width or
    // multiply/add contraction. Used only by the adversary's negative control.
    volatile float rounded = float(value);
    return rounded;
}

float same_sign_lane_negative_control(const q5_checkpoint& weights, const input_tensor& input,
                                     int row, int column) {
    // This deliberately bad stride-32 reduction is a negative control, never
    // the oracle. The retained prepack FP64 dot and signed ideal stay separate.
    std::array<float, 32> lanes{};
    for (int lane = 0; lane < 32; ++lane) {
        for (int element = lane; element < weights.inner; element += 32) {
            const float product = rounded_f32(weights.oracle[size_t(row) * weights.inner + element]
                * double(input.at(element, column)));
            lanes[size_t(lane)] = rounded_f32(double(lanes[size_t(lane)]) + double(product));
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        const auto previous = lanes;
        for (int lane = 0; lane + offset < 32; ++lane)
            lanes[size_t(lane)] = rounded_f32(double(previous[size_t(lane)])
                + double(previous[size_t(lane + offset)]));
    }
    return lanes[0];
}

std::vector<float> check_narrow_cancellation(ggml_backend_t backend, int tokens, int rows) {
    constexpr int inner = 4096;
    require(tokens > 0 && tokens <= 8, "Q5 narrow cancellation fixture requires a narrow width");
    auto* ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot create Q5 narrow cancellation context");
    q5_checkpoint weights(ctx, inner, rows, true, true, true);
    input_tensor x(ctx, GGML_TYPE_F32, {inner, tokens, 1, 1}, true, true);
    const auto role = tsg_projection_role::recurrent_decay;
    require(tsg_projection_weight(ctx, weights.tensor, role) == weights.tensor,
        "Narrow Q5 projection widened original checkpoint weights");
    auto* output = tsg_projection_quant_f32_if_required(ctx, weights.tensor, x.tensor, role);
    require(output && tsg_is_matmul_quant_f32(output) && !tsg_is_matmul_q8_f32(output)
        && output->src[0] == weights.tensor && output->src[1] == x.tensor
        && weights.tensor->type == GGML_TYPE_Q5_K && weights.tensor->nb[0] == block_bytes
        && weights.tensor->nb[1] == weights.row_bytes && x.tensor->type == GGML_TYPE_F32
        && x.tensor->nb[0] == 2 * sizeof(float) && output->type == GGML_TYPE_F32
        && output->ne[0] == rows && output->ne[1] == tokens
        && output->ne[2] == 1 && output->ne[3] == 1 && ggml_is_contiguous(output),
        "Narrow Q5 projection changed original format, strides, ownership or output layout");
    ggml_set_output(output);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "Cannot allocate Q5 narrow cancellation fixture");
    const void* weight_pointer = weights.tensor->data;
    const void* input_pointer = x.tensor->data;
    require(weight_pointer && input_pointer, "Narrow Q5 sources have no resident storage");
    for (int column = 0; column < tokens; ++column) {
        const float residual = std::ldexp(float(column + 1), -21);
        std::fill(x.values.begin() + size_t(column) * inner,
            x.values.begin() + size_t(column + 1) * inner, 1.0f);
        x.values[size_t(column) * inner] += residual;
        // A stride-32 lane grouping puts 128 same-sign unit products in
        // each lane. These residuals are strictly below half an ULP at 128,
        // so that naive grouping loses them before opposite lanes cancel.
        // Adjacent +/-1 products cancel locally before any long lane sum.
        require(x.at(0, column) - x.at(1, column) == residual
            && float(128.0 + double(residual)) == 128.0f,
            "Q5 narrow fixture does not expose same-sign lane cancellation");
    }
    weights.upload(); x.upload();
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
        "Q5 narrow cancellation graph execution failed");
    ggml_backend_synchronize(backend);
    std::vector<float> actual(size_t(rows) * tokens);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    std::vector<double> oracle(actual.size());
    for (int column = 0; column < tokens; ++column) {
        const double residual = std::ldexp(double(column + 1), -21);
        for (int row = 0; row < rows; ++row) {
            double sum = 0;
            for (int element = 0; element < inner; ++element)
                sum += weights.oracle[size_t(row) * inner + element] * double(x.at(element, column));
            const double ideal = row % 2 == 0 ? residual : -residual;
            require(sum == ideal,
                "Q5 narrow prepack FP64 dot disagrees with the independent signed residual");
            require(same_sign_lane_negative_control(weights, x, row, column) == 0.0f && ideal != 0,
                "Q5 narrow adversary failed to erase every residual under same-sign lane grouping");
            const size_t index = size_t(column) * rows + row;
            oracle[index] = sum;
            const float rounded = float(ideal);
            require(std::memcmp(&actual[index], &rounded, sizeof(float)) == 0,
                "Q5 narrow projection lost the F32 residual or changed its correctly rounded sign/value");
        }
    }
    const auto error = measure(actual, oracle);
    require(weights.tensor->data == weight_pointer && x.tensor->data == input_pointer
        && output->src[0] == weights.tensor && output->src[1] == x.tensor,
        "Narrow Q5 projection changed resident input pointers or source ownership");
    weights.check_unchanged(); x.check_unchanged();
    std::printf("SENSITIVE_Q5_NARROW_CANCELLATION backend=%s inner=%d rows=%d columns=%d full_values=%zu exact_residual_values=%zu same_sign_negative_control_zero_values=%zu max_abs=%.9g relative_l2=%.9g resident_q5=1 padded_weights=1 interleaved_input=1 prepack_fp64=1 ideal_signed_residual=1 negative_control_is_oracle=0\n",
        ggml_backend_name(backend), inner, rows, tokens, actual.size(), actual.size(), actual.size(), error.first, error.second);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return actual;
}
} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    check_format_oracle();
    // These explicit experiments change short-column reduction order. Reject
    // instead of silently rewriting the qualification environment.
    for (const char* name : {"TS_GGML_Q8_PARALLEL_VECTOR", "TS_GGML_Q8_PARALLEL_SMALL_BATCH"}) {
        const char* value = std::getenv(name);
        require(!value || std::strcmp(value, "1") != 0,
            "Clear experimental quantized parallel-reduction flags for column-identity qualification");
    }
    const char* tile = std::getenv("TS_GGML_Q8_PREFILL_TILE");
    std::printf("SENSITIVE_Q5_ENV prefill_tile=%s parallel_vector=0 parallel_small_batch=0 max_abs_gate=%.9g relative_l2_gate=%.9g\n",
        tile ? tile : "auto", maximum_error, relative_error);
#ifdef TSG_GGML_USE_CUDA
    require(!tsg_f32_projection_environment_conflicts(std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")),
        "CUDA precision override conflicts with the sensitive F32 projection contract");
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    auto* original = ggml_backend_cuda_init(0);
    require(original != nullptr, "Cannot initialize original CUDA backend");
    auto* backend = tsg_dsv4_fused_backend_init(original);
#else
    auto* backend = ggml_backend_cpu_init();
    if (backend) ggml_backend_cpu_set_n_threads(backend, 4);
#endif
    require(backend != nullptr, "Cannot initialize Q5 sensitive projection backend");
    std::vector<chain_result> results;
    for (int tokens : {1, 3, 8, 512, 774, 809}) results.push_back(check_chain(backend, tokens));
    size_t overlap_values = 0;
    const auto exact_overlap = [&overlap_values](const chain_result& narrow, const chain_result& wide) {
        const auto compare = [&overlap_values](const std::vector<float>& short_values, const std::vector<float>& wide_values) {
            require(std::memcmp(short_values.data(), wide_values.data(), short_values.size() * sizeof(float)) == 0,
                "Shared Q5 projection columns differ within a qualified width class");
            overlap_values += short_values.size();
        };
        compare(narrow.first, wide.first); compare(narrow.second, wide.second);
    };
    exact_overlap(results[0], results[2]); exact_overlap(results[1], results[2]);
    exact_overlap(results[3], results[5]); exact_overlap(results[4], results[5]);
    size_t cross_values = 0, changed_bits = 0;
    double cross_maximum = 0, cross_squared = 0, cross_reference_squared = 0;
    for (size_t index = 0; index < 3; ++index) {
        const auto observe = [&](const std::vector<float>& short_values, const std::vector<float>& wide_values) {
            for (size_t element = 0; element < short_values.size(); ++element) {
                const double delta = double(short_values[element]) - wide_values[element];
                cross_maximum = std::max(cross_maximum, std::abs(delta));
                cross_squared += delta * delta;
                cross_reference_squared += double(wide_values[element]) * wide_values[element];
                changed_bits += std::memcmp(&short_values[element], &wide_values[element], sizeof(float)) != 0;
                ++cross_values;
            }
        };
        observe(results[index].first, results[5].first); observe(results[index].second, results[5].second);
    }
    std::printf("SENSITIVE_Q5_CROSS_CLASS observations=%zu changed_bits=%zu max_abs=%.9g relative_l2=%.9g observation_only=1\n",
        cross_values, changed_bits, cross_maximum, std::sqrt(cross_squared / cross_reference_squared));
    // A full-block K=256 case also exercises contiguous Q5 rows and an F32
    // source whose elements are interleaved, in addition to the padded main cases.
    check_chain(backend, 3, 256, false, true, 129);
    // Partial vector/64-row tiles must preserve guards and all outputs too.
    // The wide case also has a partial final column tile at N=809.
    check_chain(backend, 1, 512, true, false, 129);
    check_chain(backend, 809, 512, true, false, 67);
    size_t narrow_cancellation_values = 0, narrow_cancellation_overlap_values = 0;
    for (int rows : {67, 129}) {
        std::vector<std::vector<float>> narrow_results;
        for (int tokens : {1, 3, 8}) {
            narrow_results.push_back(check_narrow_cancellation(backend, tokens, rows));
            narrow_cancellation_values += narrow_results.back().size();
        }
        for (size_t index : {size_t(0), size_t(1)}) {
            require(std::memcmp(narrow_results[index].data(), narrow_results[2].data(),
                narrow_results[index].size() * sizeof(float)) == 0,
                "Shared narrow Q5 cancellation columns differ within the 1/3/8 width class");
            narrow_cancellation_overlap_values += narrow_results[index].size();
        }
    }
    std::printf("SENSITIVE_Q5_NARROW_CANCELLATION_COVERAGE cases=6 inner=4096 row_counts=67,129 columns=1,3,8 full_values=%zu exact_residual_values=%zu within_class_bit_exact_values=%zu cross_class_bit_identity=0\n",
        narrow_cancellation_values, narrow_cancellation_values, narrow_cancellation_overlap_values);
    ggml_backend_free(backend);
#ifdef TSG_GGML_USE_CUDA
    ggml_backend_free(original);
#endif
    std::printf("PASS: Q5 sensitive two-stage complete outputs at 1/3/8/512/774/809, %zu within-class bit-exact overlap values, retained prepack FP64 oracle plus independent stored-byte/upstream format compatibility, exact F32 residual, unchanged original weights/guards, K=256/interleaved and partial129/67-row cases; K=4096 adjacent-pair cancellation checks %zu complete exact residual outputs and %zu narrow bit-exact overlap values at 1/3/8 columns and 67/129 rows; backend release; no cross-class bit identity, real-model quality/performance or hard-cap claim\n",
        overlap_values, narrow_cancellation_values, narrow_cancellation_overlap_values);
    return 0;
}
