// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Regression test: the standalone MoE FFN kernel (TSGgml_MoEFFNPrefillSwiGLUQuantF32)
// uploads its per-expert biases as graph leafs in the reusable compute buffer.
// ggml-cuda fuses {mul_mat_id, add_id, mul_mat_id, add_id, swiglu_oai} into one
// MMVQ kernel for small token batches, and that kernel reads the biases while it
// writes the activation. Its memory-overlap check skips leafs (op NONE), because
// in llama.cpp biases are weights. Here they were not: the allocator freed the gate
// bias after its add_id and placed the activation on top of it, so the fused kernel
// overwrote the bias it was still reading. gpt-oss-20b's batched paged decode of
// four sequences (TS_PER_SEQ_FUSED=0) produced different tokens from run to run.
//
// The test runs the kernel repeatedly on a small token batch and compares every
// output with an exact host evaluation of the same expert FFN. Weights and inputs
// sit on the Q8_0 / Q8_1 grids, so the only quantization error left is in the
// down projection's input and the tolerance can be tight.
//
// Metal does not fuse this chain, so its run is a plain kernel-correctness guard.
//
// Usage: GgmlOpsMoeFusedBiasAliasTest [cpu|cuda|metal]   (exit 77 = backend unavailable)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

extern "C" {
    const char* TSGgml_GetLastError();
    int TSGgml_IsBackendAvailable(int backendType);
    void TSGgml_Shutdown();
    int TSGgml_MoEFFNPrefillSwiGLUQuantF32(
        float* hidden_in, float* hidden_out, int seq_len, int hidden_dim, int n_ff,
        int num_experts, int n_used, const std::int32_t* selected_experts, const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_total_bytes,
        void* up_data, int up_type, std::int64_t up_ne0, std::int64_t up_ne1, std::int64_t up_total_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_total_bytes,
        const float* gate_bias, const float* up_bias, const float* down_bias,
        int activation_type, float oai_alpha, float oai_limit, int run_on_cpu);
}

namespace
{
    constexpr int BackendTypeMetal = 1;
    constexpr int BackendTypeCpu = 2;
    constexpr int BackendTypeCuda = 3;
    constexpr int GgmlTypeQ8_0 = 8;
    constexpr int ActivationSwiGluOai = 1;
    constexpr int Block = 32;
    constexpr int BlockBytes = 34; // f16 scale + 32 int8 quants

    // Dimensions shaped like gpt-oss's decode: more experts than (tokens x used
    // experts), so the activation fits inside a freed per-expert bias. Token
    // counts 1..7 are the ones ggml-cuda serves with the fused MMVQ kernels on
    // this quant type (one token and the multi-token MoE kernel).
    constexpr int Hidden = 256;
    constexpr int Ff = 256;
    constexpr int Experts = 32;
    constexpr int Used = 4;
    constexpr float Alpha = 1.702f;
    constexpr float Limit = 7.0f;

    [[noreturn]] void fail(const std::string& message)
    {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }

    std::uint32_t g_rng = 0x9e3779b9u;
    std::uint32_t next_u32()
    {
        g_rng ^= g_rng << 13;
        g_rng ^= g_rng >> 17;
        g_rng ^= g_rng << 5;
        return g_rng;
    }
    int next_int(int lo, int hi) { return lo + static_cast<int>(next_u32() % static_cast<std::uint32_t>(hi - lo + 1)); }

    // Q8_0 weights with power-of-two scales, which f16 stores exactly, so the host
    // reference dequantizes to exactly the values the kernel multiplies with.
    struct QuantMatrix
    {
        int rows = 0;
        int cols = 0;
        std::vector<std::uint8_t> bytes;  // [Experts][rows][cols/Block blocks]
        std::vector<double> values;       // dequantized, same layout

        QuantMatrix(int r, int c) : rows(r), cols(c)
        {
            const int blocks = c / Block;
            bytes.resize(static_cast<std::size_t>(Experts) * r * blocks * BlockBytes);
            values.resize(static_cast<std::size_t>(Experts) * r * c);
            static constexpr std::uint16_t half_scales[] = { 0x2000, 0x2400, 0x2800 }; // 2^-7, 2^-6, 2^-5
            static constexpr double scales[] = { 1.0 / 128.0, 1.0 / 64.0, 1.0 / 32.0 };
            std::size_t b = 0;
            for (int e = 0; e < Experts; ++e)
                for (int row = 0; row < r; ++row)
                    for (int k = 0; k < blocks; ++k)
                    {
                        const int s = next_int(0, 2);
                        bytes[b] = static_cast<std::uint8_t>(half_scales[s] & 0xff);
                        bytes[b + 1] = static_cast<std::uint8_t>(half_scales[s] >> 8);
                        for (int i = 0; i < Block; ++i)
                        {
                            const int q = next_int(-127, 127);
                            bytes[b + 2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
                            values[(static_cast<std::size_t>(e) * r + row) * c + k * Block + i] = q * scales[s];
                        }
                        b += BlockBytes;
                    }
        }

        double dot(int expert, int row, const double* x) const
        {
            const double* w = &values[(static_cast<std::size_t>(expert) * rows + row) * cols];
            double sum = 0.0;
            for (int i = 0; i < cols; ++i) sum += w[i] * x[i];
            return sum;
        }
    };

    std::vector<double> reference(
        int Tokens, const std::vector<float>& input, const std::vector<std::int32_t>& ids, const std::vector<float>& routing,
        const QuantMatrix& gate, const QuantMatrix& up, const QuantMatrix& down,
        const std::vector<float>& gate_bias, const std::vector<float>& up_bias, const std::vector<float>& down_bias)
    {
        std::vector<double> out(static_cast<std::size_t>(Tokens) * Hidden, 0.0);
        std::vector<double> x(Hidden), act(Ff);
        for (int t = 0; t < Tokens; ++t)
        {
            for (int i = 0; i < Hidden; ++i) x[i] = input[static_cast<std::size_t>(t) * Hidden + i];
            for (int u = 0; u < Used; ++u)
            {
                const int e = ids[static_cast<std::size_t>(t) * Used + u];
                for (int r = 0; r < Ff; ++r)
                {
                    double g = gate.dot(e, r, x.data()) + gate_bias[static_cast<std::size_t>(e) * Ff + r];
                    double v = up.dot(e, r, x.data()) + up_bias[static_cast<std::size_t>(e) * Ff + r];
                    g = std::min(g, static_cast<double>(Limit));
                    v = std::max(std::min(v, static_cast<double>(Limit)), -static_cast<double>(Limit));
                    act[r] = g / (1.0 + std::exp(-g * Alpha)) * (1.0 + v);
                }
                const double w = routing[static_cast<std::size_t>(t) * Used + u];
                for (int h = 0; h < Hidden; ++h)
                    out[static_cast<std::size_t>(t) * Hidden + h] +=
                        w * (down.dot(e, h, act.data()) + down_bias[static_cast<std::size_t>(e) * Hidden + h]);
            }
        }
        return out;
    }
}

struct Model
{
    QuantMatrix gate{Ff, Hidden}, up{Ff, Hidden}, down{Hidden, Ff};
    std::vector<float> gate_bias, up_bias, down_bias;
};

// Returns false (after printing why) when the kernel disagrees with the host or with itself.
bool run_case(const std::string& name, Model& m, int Tokens)
{
    auto& gate = m.gate; auto& up = m.up; auto& down = m.down;
    const auto& gate_bias = m.gate_bias; const auto& up_bias = m.up_bias; const auto& down_bias = m.down_bias;

    // Inputs on the Q8_1 grid: every block holds +/-127/16, so its scale is 1/16 exactly.
    std::vector<float> input(static_cast<std::size_t>(Tokens) * Hidden);
    for (int t = 0; t < Tokens; ++t)
        for (int i = 0; i < Hidden; ++i)
            input[static_cast<std::size_t>(t) * Hidden + i] =
                (i % Block == 0 ? (next_u32() & 1 ? 127 : -127) : next_int(-127, 127)) / 16.0f;

    // Distinct experts per token, spread over the whole range.
    std::vector<std::int32_t> ids(static_cast<std::size_t>(Tokens) * Used);
    std::vector<float> routing(static_cast<std::size_t>(Tokens) * Used);
    for (int t = 0; t < Tokens; ++t)
        for (int u = 0; u < Used; ++u)
        {
            ids[static_cast<std::size_t>(t) * Used + u] = (t * 7 + u * 5 + 3) % Experts;
            routing[static_cast<std::size_t>(t) * Used + u] = 0.25f + 0.125f * ((t + u) % 3);
        }

    const std::vector<double> expected = reference(Tokens, input, ids, routing, gate, up, down, gate_bias, up_bias, down_bias);
    double scale = 0.0;
    for (double v : expected) scale = std::max(scale, std::fabs(v));

    constexpr int Repeats = 64;
    double worst = 0.0;
    int worst_repeat = -1, worst_index = -1;
    std::vector<float> first;
    bool deterministic = true;
    for (int repeat = 0; repeat < Repeats; ++repeat)
    {
        std::vector<float> hidden = input;
        std::vector<float> out(static_cast<std::size_t>(Tokens) * Hidden, 0.0f);
        const int ok = TSGgml_MoEFFNPrefillSwiGLUQuantF32(
            hidden.data(), out.data(), Tokens, Hidden, Ff, Experts, Used, ids.data(), routing.data(),
            gate.bytes.data(), GgmlTypeQ8_0, Hidden, Ff, static_cast<std::int64_t>(gate.bytes.size()),
            up.bytes.data(), GgmlTypeQ8_0, Hidden, Ff, static_cast<std::int64_t>(up.bytes.size()),
            down.bytes.data(), GgmlTypeQ8_0, Ff, Hidden, static_cast<std::int64_t>(down.bytes.size()),
            gate_bias.data(), up_bias.data(), down_bias.data(),
            ActivationSwiGluOai, Alpha, Limit, /*run_on_cpu=*/0);
        if (ok == 0)
        {
            const char* err = TSGgml_GetLastError();
            fail(std::string("kernel call failed: ") + (err != nullptr ? err : "<null>"));
        }
        if (repeat == 0) first = out;
        else if (std::memcmp(first.data(), out.data(), out.size() * sizeof(float)) != 0) deterministic = false;
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            const double err = std::fabs(out[i] - expected[i]);
            if (!(err <= worst)) { worst = err; worst_repeat = repeat; worst_index = static_cast<int>(i); }
        }
    }

    // The down projection quantizes its (off-grid) input to Q8_1; everything else is exact.
    const double tolerance = 0.01 * std::max(1.0, scale);
    std::printf("%s tokens=%d: %d repeats, output scale %.4f, worst |error| %.6g (repeat %d, token %d, dim %d), tolerance %.6g, deterministic=%s\n",
        name.c_str(), Tokens, Repeats, scale, worst, worst_repeat, worst_index / Hidden, worst_index % Hidden, tolerance,
        deterministic ? "yes" : "no");
    bool ok = true;
    if (!(worst <= tolerance))
    {
        std::printf("  FAIL: output differs from the host evaluation (a fused kernel read a bias the graph had already overwritten?)\n");
        ok = false;
    }
    if (!deterministic)
    {
        std::printf("  FAIL: repeated calls on identical input returned different outputs\n");
        ok = false;
    }
    return ok;
}

int main(int argc, char** argv)
{
    int backend = BackendTypeCpu;
    std::string name = "cpu";
    if (argc >= 2)
    {
        name = argv[1];
        if (name == "cuda" || name == "ggml_cuda") backend = BackendTypeCuda;
        else if (name == "metal" || name == "ggml_metal") backend = BackendTypeMetal;
        else if (name != "cpu" && name != "ggml_cpu") fail("unknown backend argument " + name + "; use cpu, cuda or metal");
    }
    if (TSGgml_IsBackendAvailable(backend) == 0)
    {
        const char* err = TSGgml_GetLastError();
        std::cout << "SKIP: " << name << " backend unavailable: " << (err != nullptr ? err : "<null>") << std::endl;
        return 77;
    }

    Model m;
    // Biases large enough that reading an activation in place of one is visible.
    m.gate_bias.resize(static_cast<std::size_t>(Experts) * Ff);
    m.up_bias.resize(static_cast<std::size_t>(Experts) * Ff);
    m.down_bias.resize(static_cast<std::size_t>(Experts) * Hidden);
    for (float& v : m.gate_bias) v = next_int(-48, 48) / 16.0f;
    for (float& v : m.up_bias) v = next_int(-48, 48) / 16.0f;
    for (float& v : m.down_bias) v = next_int(-48, 48) / 16.0f;

    bool ok = true;
    for (int tokens : { 1, 4, 7 })
        ok = run_case(name, m, tokens) && ok;
    // Release the cached buffers and the backend before exit: ggml-metal asserts in
    // its static destructor while residency sets still hold buffers.
    TSGgml_Shutdown();
    if (!ok)
        fail("the MoE FFN kernel disagreed with the host evaluation or with itself");
    std::printf("PASS\n");
    return 0;
}
