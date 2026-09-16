// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <cstdint>

// Decode-class width for the TensorSharp-owned F32 precision paths (explicit
// F32 matmul and F32 attention on CUDA).
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
// drafter's verify width exceeds it.
constexpr int64_t TSG_PRECISION_DECODE_COLUMNS = 8;
