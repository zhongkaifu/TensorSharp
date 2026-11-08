// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// TensorSharp Metal (tsm): a tiered MoE layer's accelerator half - the routes its slots hold - computed by the grouped
// expert kernels ported from Strata (tsm/kernels, MIT, tsm/LICENSE-STRATA) instead of ggml-metal's MUL_MAT_ID. The work
// is encoded on ggml-metal's own command queue at the layer's host seam, so it runs after the segment that produced
// the layer's input and before the segment that reads its output, without a host round trip of its own: the queue
// orders it, and ggml-metal's buffers are hazard-tracked.
//
// Each distinct slot's gate/up/down rows are read once and applied to every row of the window that routes to it
// (ggml-metal reads them once per (row, route)): on an M5 Pro, ISTA Q2_0, the experts of a 1-row step cost 6.9 ms of
// GPU time through MUL_MAT_ID (89 GB/s) and 16.2 ms in a 4-row verify window.
#pragma once

#include <cstdint>

struct ggml_tensor;
struct ggml_backend;

namespace tsg { struct MoeHotTier; }

namespace tsm
{
    // TS_METAL_MOE=0 keeps ggml's MUL_MAT_ID for the tier's routes.
    bool moe_enabled();

    // Whether the kernels take this tier's formats and shapes for a window of T rows on this backend.
    bool moe_supported(ggml_backend* backend, const tsg::MoeHotTier& tier, int hidden, int n_ff, int T, int n_used);

    // Encodes (and commits, without waiting) out[t] = sum over the held routes of row t, in route order, of
    // weight * down(silu(gate x_t) * up x_t), x = moe_in [hidden, T], out = [hidden, T] F32. slots[i] is route i's tier
    // slot, or -1 for a route the host computes. Both tensors must live in ggml-metal buffers. False with
    // set_last_error on a failure; nothing is committed then.
    bool moe_encode(ggml_backend* backend, const tsg::MoeHotTier& tier, const ggml_tensor* moe_in, ggml_tensor* out,
                    int hidden, int n_ff, int T, int n_used, const std::int32_t* slots, const float* weights);

    // ---- Seams on a shared event (Strata's host turns, metal/src/mtl_rt.mm run_graph_seg, over ggml's queue).
    //
    // A seamed pass otherwise ends each segment with a submit and a wait, and the accelerator idles while the host
    // wakes, reads the routes, encodes the held routes and the next segment and commits them. Here the segments are
    // enqueued ahead and each seam is a command buffer of its own on ggml-metal's queue:
    //
    //   xprep (reads moe_in: hazard tracking orders it after the segment) | signal A | wait B | moe_gu, moe_down
    //   (grids from the plan) | combine (writes `out`, which the next segment reads: orders it after)
    //
    // The host spins until A, reads the routes, computes the ones no slot holds into moe_out, writes the plan and
    // signals B. Same kernels, same order, same buffers as moe_encode: bit-identical.
    // TS_METAL_SEAM_EVENTS=0 keeps a submit and a wait per seam.
    bool seams_enabled(ggml_backend* backend);
    // Whether the host can read and write t in place (a shared ggml-metal buffer): a seam's host turn must not need a
    // blit, which would queue behind the pass's own waits.
    bool seams_host_visible(const ggml_tensor* t);
    // Starts a pass of n_seams seams on this thread; false (with the last error) when the event cannot be made.
    bool seams_begin(ggml_backend* backend, int n_seams);
    // Whether this thread is inside a seamed pass (the route split writes the plan instead of encoding).
    bool seams_active();
    // Enqueues seam `seam`'s command buffer, right after the segment that ends at it.
    bool seams_enqueue(int seam, const tsg::MoeHotTier& tier, const ggml_tensor* moe_in, const ggml_tensor* sel_ids,
                       const ggml_tensor* weights, ggml_tensor* out, int hidden, int n_ff, int T, int n_used);
    // Spins until the accelerator has finished the segment before `seam`, whose host turn then runs; false when a
    // command buffer failed.
    bool seams_wait(int seam);
    // The current seam's plan (slots[i] = route i's tier slot, or -1), staged until the host releases the seam.
    void seams_write_plan(int hidden, int n_ff, int T, int n_used, const std::int32_t* slots, const float* weights);
    // The current seam's host routes are in moe_out: its plan is written and its kernels, and then the next segment,
    // run.
    void seams_release_out();
    // Ends the pass. `failed`: releases every wait still pending so the queue drains (the results are void).
    void seams_end(bool failed);
    // Passes that ran to the end in this process (a test proves the path engaged).
    long long seams_pass_count();
}
