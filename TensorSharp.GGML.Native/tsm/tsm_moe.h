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

    // ---- A GDN layer's recurrence for a prompt chunk (tsm/kernels/tsm_gdn_prompt.metal, ported from Strata's
    // gdn_prompt.metal): conv + SiLU + q/k L2 norms + gates, the state scan, the gated RMS norm - three dispatches in
    // place of ggml's transpose, concat, ssm_conv, q/k/v copies, head repeat, gated delta net and norm. The span cuts
    // its graph after the layer's input projections and encodes these on ggml-metal's queue between the two segments
    // (hazard tracking orders them); the conv history and the recurrent state (ggml's layout) are updated in place.
    // TS_METAL_GDN_PROMPT=0 keeps ggml's ops.
    struct GdnPromptIO
    {
        const ggml_tensor* qkv = nullptr;      // [channels, T]: q | k | v
        const ggml_tensor* z = nullptr;        // [h_v * 128, T]
        const ggml_tensor* alpha = nullptr;    // [h_v, T]
        const ggml_tensor* beta = nullptr;     // [h_v, T]
        const ggml_tensor* conv_w = nullptr;   // [4, channels]
        const ggml_tensor* dt = nullptr;       // [h_v]
        const ggml_tensor* a = nullptr;        // [h_v]
        const ggml_tensor* norm_w = nullptr;   // [128]
        ggml_tensor* conv_state = nullptr;     // [3, channels], updated in place
        ggml_tensor* ssm_state = nullptr;      // [128, 128, h_v], updated in place
        ggml_tensor* y = nullptr;              // [h_v * 128, T], written
        int T = 0, h_k = 0, h_v = 0, channels = 0;
        float eps = 0.0f;
    };
    // Whether the kernels take this layer's shapes (128-wide heads, a 4-tap conv) on this backend.
    bool gdn_prompt_supported(ggml_backend* backend, int head_k_dim, int head_v_dim, int h_k, int h_v, int d_conv);
    // Encodes and commits (without waiting) the layer's three passes. False with set_last_error on a failure.
    bool gdn_prompt_encode(ggml_backend* backend, const GdnPromptIO& io);
    // Prompt chunks whose GDN layers ran on these kernels (a test proves the path engaged).
    long long gdn_prompt_count();

    // ---- A prompt chunk's hyper-connection element-wise passes (tsm/kernels/tsm_hc_prompt.metal, after Strata's
    // gr_epi_up / gr_norm_h): one pass each in place of ggml's chains, encoded between graph segments like the GDN
    // kernels. Tensors F32 and contiguous. TS_METAL_HC_PROMPT=0 keeps ggml's ops.
    bool hc_prompt_supported(ggml_backend* backend);
    // mixed [n, T] = (sum over the hc streams of xn * sigmoid(G)) / hc; xn and G [hc * n, T].
    bool hc_mix_encode(ggml_backend* backend, const ggml_tensor* xn, const ggml_tensor* G, ggml_tensor* mixed,
                       int n_embd, int hc, int T);
    // out [hc * n, T] = res + b (x) 2 sigmoid(inject / hc); b [n, T], inject [hc, T]; out may be res.
    bool hc_scatter_encode(ggml_backend* backend, const ggml_tensor* res, const ggml_tensor* b, const ggml_tensor* inject,
                           ggml_tensor* out, int n_embd, int hc, int T);
    // Hyper-connection passes that ran on these kernels (a test proves the path engaged).
    long long hc_prompt_count();

    // ---- A QSA layer's cell selection for a prompt chunk (tsm/kernels/tsm_qsa_prompt.metal): the per-token block
    // scores (relu, head sum, bias) and a radix select of the `width` best allowed cells, written as the attention
    // mask, in place of ggml's gather of every cell's score, two transposes and a top-k sort over [cells, T].
    // scores F32 [B, H, T], bias F32 [B, T], cell_blocks I32 [cells], masks F16 [cells, T]. Cutoff ties are taken
    // lowest cell first. TS_METAL_QSA_SELECT=0 keeps ggml's ops.
    bool qsa_select_supported(ggml_backend* backend);
    bool qsa_select_encode(ggml_backend* backend, const ggml_tensor* scores, const ggml_tensor* bias,
                           const ggml_tensor* cell_blocks, const ggml_tensor* mask_in, ggml_tensor* mask_out,
                           int T, int n_cells, int n_blocks, int heads, int width);
    // Selections that ran on these kernels (a test proves the path engaged).
    long long qsa_select_count();
}
