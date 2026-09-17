// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <cstdint>
#include <cstdlib>

// Decode-class width for the TensorSharp-owned F32 precision paths (explicit
// F32 matmul on CUDA and F32 attention on CPU/CUDA).
//
// A launch with at most this many activation columns / query rows must compute
// every column with exactly the kernel, key partition and reduction order a
// single-column launch would use. Above it the paths may switch to throughput
// implementations (cuBLAS SGEMM, tiled SGEMM attention) whose rounding differs
// at the last F32 bit.
//
// Why bit-for-bit matters here: DeepSeek V4.1 quantizes what these paths
// produce (raw/compressed K rings and the DSpark drafter's key rings) and its
// sparse routing bins them. A last-bit difference in a pre-quantization value
// that sits on a grid boundary becomes a whole quantization step in the
// committed cache row. A speculative verify pass commits block_size + 1 rows
// per call; if those rows were computed by a different kernel than the
// single-token decode step that would otherwise have produced them, a rewind
// to the accepted prefix leaves state that differs from a plain decode of the
// same tokens, and the drafter's next confidence/tokens diverge from greedy.
// The bound therefore has to cover the widest speculative verify (DSpark's
// block size is 5, so 6 rows) and the ordinary forwards of the same width
// that the parity fixtures compare it against; 8 also matches the batch the
// upstream mul_mat_vec kernels treat as decode. dsv4_load warns when a
// drafter's verify width exceeds it; wider verification batches require
// independent numerical and rewind qualification.
//
// Cost: the invariant paths give up the throughput kernels at these widths. A
// verify-width attention launch (<= 8 queries, 16 fixed key splits) measured
// 1.8-2.1x slower per launch than the tiled SGEMM path at 8192 keys on an A40;
// single-query decode already used the split-key path and is unchanged. The
// owned aarch64 CPU dot that keeps neighbouring F32 products together measured
// within 0.76% of ggml's upstream dot (numerical-r3 ARM benchmark).
//
// Qwen 3.8 Flash Next (ggml_ops_qwen4exp.cpp, Q4eRowKernels) uses the same
// bound for the same reason without owned kernels: a CUDA span graph of up to
// this many tokens builds each row from the kernels its one-token graph would
// run (float projections on the broadcast axis, experts and attention one row
// at a time), so a verify block commits exactly what plain decoding would.
constexpr int64_t TSG_PRECISION_DECODE_COLUMNS = 8;

// Smallest attention key extent (raw ring plus visible compressed rows) at
// which DeepSeek V4.1 prefill hands the owned F32 attention a sparse capacity.
constexpr int64_t TSG_DSV41_SPARSE_MIN_KEYS = 8192;

// Sparse capacity DeepSeek V4.1's owned F32 attention (the default CUDA path,
// tsg_attention_f32_on_backend) gets for one launch; 0 selects the dense
// kernels. `sparse_env` is TS_DSV41_SPARSE_FA as read from the environment.
//
// A V4.1 query sees at most its sliding window plus the indexer's selected
// compressed rows, so `n_swa + indexer_top_k` keys bound every row. The mask
// compaction attends to exactly those keys, in key order, and a row with more
// finite entries than the capacity takes the full-row fallback, so the bound
// can never drop a visible key.
//
// Why sparse is the default at long keys: above the decode-class width the
// dense path is tiled SGEMM attention, whose score budget drops to 96 MiB at
// 8,192 keys and more, so its cost grows about quadratically with the keys.
// Measured on an A40 with 512 queries, 64 heads, 640 visible keys per query
// (GgmlOpsCudaAttentionPrecisionTest --benchmark-dsv41-prefill): tiled 106.8 /
// 1,547-1,549 / 6,864.6 ms at 8,960 / 33,536 / 66,304 keys, compacted 34.4 /
// 34.1-34.3 / 34.8 ms. Against a decomposed F32 reference the compacted kernel
// stays within max_abs 1.5e-7 (rel_l2 7.5e-7), the tiled one 8.9e-8 (4.8e-7).
// Below 8,192 keys every launch keeps the dense kernels, so short prompts are
// unchanged bit for bit.
//
// Why only above TSG_PRECISION_DECODE_COLUMNS queries: a DSpark verify (six
// rows) must keep the split-key kernel single-token decode uses, or the rows
// it commits differ from what decode would have written. A sparse query
// differed from the same query through the decode kernel in 31,198 of 32,768
// outputs (CUDA, 8,960 keys; attention_precision_test's sparse invariance
// check), while the decode kernel is bit-identical from 1 to 8 queries. A
// 6-query verify over 33,536 keys therefore keeps its dense launch: 9.2 ms,
// the same with the variable unset or 0.
//
// TS_DSV41_SPARSE_FA=0 (any value atoi reads as 0) restores tiled prefill, the
// previous default. Unset or any other value selects the sparse kernel; "1"
// additionally opts the non-owned ggml flash-attention branch into its sparse
// hint, which keeps its own gate and F16 internals (see ggml_ops_deepseek4.cpp).
inline int tsg_dsv41_owned_sparse_capacity(int64_t queries, int64_t n_kv, int64_t n_swa, int64_t indexer_top_k,
                                           const char * sparse_env)
{
    if (sparse_env && std::atoi(sparse_env) == 0) return 0;
    if (queries <= TSG_PRECISION_DECODE_COLUMNS || n_kv < TSG_DSV41_SPARSE_MIN_KEYS) return 0;
    if (n_swa < 0 || indexer_top_k < 0 || n_swa + indexer_top_k <= 0) return 0;
    return int(n_swa + indexer_top_k);
}
