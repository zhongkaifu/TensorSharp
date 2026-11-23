// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Inkling 2's fused stages on ggml-cuda (GraphFusion.Inkling*): GGML_OP_CUSTOM
// nodes whose tsg_dsv4_fused_desc names one of the kinds below and carries
// tsg_inkling_cuda_compute as its cuda_compute hook, so the TensorSharp
// fused-op backend (ggml_ops_dsv4_fused.cu) launches them on the ggml-cuda
// stream between the graph's ordinary runs. The graph builder makes them
// (ggml_ops_graph_builder.cpp, try_serve_inkling); the vendored ggml is used
// through its public API only.
// ---------------------------------------------------------------------------
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdint>

enum tsg_inkling_kind : int32_t
{
    // x [d, heads, rows][, w [d, heads]] -> [d, heads, rows]. f0 threshold.
    TSG_INKLING_CLIP = 40,
    // x [channels, rows], window [taps - 1, channels] (written), kernel [taps, channels][, dyn [taps * heads, rows]]
    // -> [channels, rows]. i0 heads (0: none), f0 scale.
    TSG_INKLING_CONV = 41,
    // states [hidden, n, rows] (strided), query [hidden, queries] -> [hidden, queries, rows]. f0 eps, f1 scale.
    TSG_INKLING_MIX = 42,
    // x, m [hidden, rows], down [2 hidden, rank], gain, forget [rank, hidden] -> [hidden, rows + workspace rows]
    // (the result is its first rows). f0 eps.
    TSG_INKLING_CONTROL = 43,
    // Inkling 2's attention over one sequence's keys, the keys a row sees found from the positions (no mask, no distance
    // index), each score's relative bias gathered from the row's table:
    //   q [d, rows, heads] (F32, rows and heads at any strides), k [d, keys, kv], v [dv, keys, kv] (F16 or F32, keys and
    //   heads at any strides), T [extent + 1, heads, rows] (F32 contiguous: B r - t by distance, the last entry - t),
    //   positions (I32: the step's rows' positions) -> [dv, heads, rows].
    // A row at position p sees the key at position kp when kp >= floor, kp <= p and (band == 0 or p - kp < band); its
    // logit is q·k * scale + T[min(p - kp, extent)], and a zero logit with a zero value (the sink) closes every row's
    // softmax. Key j's position: i0 > 0 (a ring of i0 cells) the last position the cell held by the rows' last position,
    // else i3 + j. i1 extent, i2 band (0: none), i4 the attention floor, i5 the rows' first index in positions; f0 the
    // scale. Up to 16 rows run split over the keys (CUDA cores); more on tensor cores (128-wide heads, F16 operands).
    TSG_INKLING_ATTENTION = 44,
    // logits [E, rows], bias [E] -> ids [K, rows] (I32). i0 K.
    TSG_INKLING_ROUTE_IDS = 45,
    // logits [E, rows], ids [K, rows] -> weights [1, K, rows]. i0 K, f0 scale.
    TSG_INKLING_ROUTE_WEIGHTS = 46,
    // gate|up [2 ff, K, rows] -> [ff, K, rows]. f0 beta, f1 linear beta (0: none).
    TSG_INKLING_SITU = 47,
    // down [hidden, K, rows], weights [1, K, rows] -> [hidden, rows] (the products added in expert order).
    TSG_INKLING_EXPERT_SUM = 48,
    // experts [K, M, E] (NVFP4, Q8_0 or IQ4_NL), x [K, 1 or used, rows] (F32), ids [used, rows] (I32) -> [M, used, rows], as
    // MUL_MAT_ID, with its workspace (the pairs grouped by expert, the activations in int8) in the node's tail.
    TSG_INKLING_EXPERTS = 49,
    // x [channels, rows], kernel [taps, channels][, dyn [taps * heads, rows]] -> [channels, rows]: several sequences' short
    // convolutions; the sequences' state leaves in the desc's object (tsg_inkling_conv_batch). i0 heads, i1 verify, f0 scale.
    TSG_INKLING_CONV_BATCH = 50,
    // base [padded, lanes] (strided columns), fold [padded, lanes], bias [padded] -> [lanes, 3]: each lane's pick (as F32),
    // the softcapped softmax's probability there, the pick again. i0 vocab, f0 softcap.
    TSG_INKLING_PICK = 51,
    // pick [lanes, 3] -> [lanes] (I32): the first row as indices.
    TSG_INKLING_PICK_I32 = 52,
    // w [rel, extent + 1], r [rel, heads, rows] (row strides) -> [extent + 1, heads, rows] (F16): the relative-bias table.
    TSG_INKLING_REL_TABLE = 53,
};

// A batched convolution's sequences: rows, first row, window leaf and (verify) history leaf each; the channels' offset in
// the leaves and taps - 1.
#define TSG_INKLING_CONV_BATCH_MAX 16
struct tsg_inkling_conv_batch
{
    int n = 0, offset = 0, hist = 0;
    int first[TSG_INKLING_CONV_BATCH_MAX] = {}, count[TSG_INKLING_CONV_BATCH_MAX] = {};
    struct ggml_tensor * window[TSG_INKLING_CONV_BATCH_MAX] = {};
    struct ggml_tensor * history[TSG_INKLING_CONV_BATCH_MAX] = {};
};

// Bytes of workspace an experts node needs past its result.
int64_t tsg_inkling_experts_workspace_bytes(int64_t k, int64_t experts, int64_t used, int64_t rows, int64_t activations);

// Whether the experts kernel takes a weight type.
bool tsg_inkling_experts_supported(enum ggml_type type);

// Whether a prompt chunk's experts product (more than 32 rows) runs on FP4 tensor cores: NVFP4 weights, k and m multiples
// of 128, a library built for an architecture-specific target that has the FP4 MMA (probed once), and TS_INKLING_EXPERTS_FP4 not 0.
bool tsg_inkling_experts_fp4(enum ggml_type type, int64_t k, int64_t m);

// Bytes of workspace the FP4 path needs past the result.
int64_t tsg_inkling_experts_fp4_workspace_bytes(int64_t k, int64_t experts, int64_t used, int64_t rows, int64_t activations);

#ifdef TSG_GGML_USE_CUDA
// Launch an Inkling node on cuda_backend's stream (the desc's cuda_compute hook).
void tsg_inkling_cuda_compute(struct ggml_tensor * dst, ggml_backend_t cuda_backend);

// Floats of workspace an attention node needs past its result (the split kernel's partials).
int64_t tsg_inkling_attention_workspace_floats(int64_t rows, int64_t heads, int64_t kv_heads, int64_t keys, int64_t dv);

// Whether the attention kernels take these shapes (head widths, heads per KV head).
bool tsg_inkling_attention_supported(int d, int dv, int heads, int kv_heads);

// A replayable graph's whole run as one CUDA graph (unless TS_INKLING_CAPTURE=0): `warmup` runs go to the fused-op
// backend as they are, the next is captured on cuda_backend's stream and every later one replays it; ggml-cuda's own
// capture is off for the runs it is handed (tsg_cuda_disable_graphs). The fused-op backend otherwise submits a decode step as many
// ggml-cuda runs split at the fused nodes, which the host takes longer to issue than the device to run. A capture that
// fails leaves the graph running as it is. The state is the graph's (made by _new, freed by _free).
void * tsg_inkling_capture_new();
void tsg_inkling_capture_free(void * capture);
enum ggml_status tsg_inkling_compute(void * capture, ggml_backend_t fused_backend, ggml_backend_t cuda_backend,
                                     struct ggml_cgraph * graph, int warmup);

// TS_INKLING_EVENT_PROFILE=1 (diagnostics): the capture puts an event record node before every graph node (each node its own
// ggml-cuda run), and every replay is timed node by node on the device, as the production replay runs (no host gaps, no
// synchronize per node, but a synchronize after each replay). Costs are summed per op (matrix products by weight type,
// shape and rows; fused nodes by name) and printed every TS_INKLING_EVENT_PROFILE_EVERY graphs (default 32) per graph
// size. tsg_inkling_event_profile captures, runs and releases a graph that is not replayed.
bool tsg_inkling_event_profile_enabled();
enum ggml_status tsg_inkling_event_profile(ggml_backend_t fused_backend, ggml_backend_t cuda_backend, struct ggml_cgraph * graph);
#endif
