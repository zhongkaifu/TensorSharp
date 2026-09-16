// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tsg_dsv41_tp
{
struct source
{
    std::string path;
    size_t offset = 0;
    ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, 4> ne = {1, 1, 1, 1};
};

struct strip
{
    int64_t first = 0;
    int64_t count = 0;
};

// A contiguous, complete tiling in down-projection quantization blocks.
// Rotating the extra blocks by layer balances V4.1's nine Q2_K blocks.
// Unquantized strips also honor ggml CUDA's two-element vector alignment.
std::vector<strip> split(int64_t width, int64_t block, int ranks, int layer);

// Prefer 64-channel floating-point strips where possible. This keeps BF16/F16
// gate/up rows and down inner dimensions eligible for the same matrix path as
// the full tensor, avoiding shape-dependent activation rounding. Use this for
// both upload layout and loader memory pricing. Quantized/F32 layouts retain
// their original block policy; small floating tensors retain vector alignment.
std::vector<strip> split_weights(int64_t width, ggml_type down_type, int ranks, int layer);

class executor
{
public:
    // Each rank owns a separate backend queue for the supplied device. CPU
    // devices are accepted for the independent numerical regression test.
    executor(const std::vector<ggml_backend_dev_t> & devices, int used_experts);
    ~executor();
    executor(const executor &) = delete;
    executor & operator=(const executor &) = delete;

    // Load only each rank's column-parallel gate/up and row-parallel down
    // strips. No full expert tensor is allocated on any participating rank.
    void add_layer(int layer, const source & gate, const source & up,
                   const source & down, float clamp_limit);
    bool has_layer(int layer) const;
    size_t rank_weight_bytes(int rank) const;

    // F32 x[embedding,tokens], routing weights[1,used,tokens], and I32 expert
    // IDs[used,tokens] -> routed output[embedding,tokens]. Pin this boundary
    // node to CPU; ordinary attention/shared-expert placement stays unchanged.
    ggml_tensor * build(ggml_context * ctx, int layer, ggml_tensor * x,
                        ggml_tensor * weights, ggml_tensor * ids);

    // Native graph callbacks cannot return a status. They fill NaNs on error;
    // callers must inspect this after graph execution and refuse generation.
    // Start once per validated request, never once per layer: a later healthy
    // layer must not conceal an earlier rank failure in the same request.
    void begin_forward();
    std::string error() const;

#if defined(TSG_GGML_TEST_HOOKS)
    void test_set_position(int64_t position);
    void test_single_fanout(bool enabled);
#endif

private:
    struct impl;
    std::unique_ptr<impl> state;
};
}
