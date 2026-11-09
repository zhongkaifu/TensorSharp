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
    // mask_out (F16 [cells, T]) and / or idx (I32 [width, T], each token's chosen cells ascending) with cnt (I32 [T]);
    // either may be null.
    bool qsa_select_supported(ggml_backend* backend);
    bool qsa_select_encode(ggml_backend* backend, const ggml_tensor* scores, const ggml_tensor* bias,
                           const ggml_tensor* cell_blocks, const ggml_tensor* mask_in, ggml_tensor* mask_out,
                           ggml_tensor* idx, ggml_tensor* cnt, int T, int n_cells, int n_blocks, int heads, int width);
    // Selections that ran on these kernels (a test proves the path engaged).
    long long qsa_select_count();

    // ---- A QSA layer's attention for a prompt chunk over each token's chosen cells only (tsm_qsa_attn), instead of
    // every cell of the window: one threadgroup per (token, kv head), the kv head's query heads as the rows of the
    // products. q F32 [D, T, H]; k, v the F16 caches [D, cells, kv heads] (any row / head strides); idx and cnt from
    // qsa_select_encode; mask_in F16 [cells, T] (each chosen cell's own mask value is added); out F32 [D, H, T] (ggml's
    // flash-attention layout). D = 256, up to 16 query heads a kv head. TS_METAL_QSA_ATTN=0 keeps ggml's flash
    // attention; TS_METAL_QSA_ATTN_MIN_CELLS (default 8192 on the tensor API, 12288 otherwise) is the narrowest window.
    bool qsa_attn_supported(ggml_backend* backend, int head_dim, int n_head, int n_head_kv, int kv_type, int n_cells);
    // Whether it runs on the Metal 4 tensor API here (tsm_qsa_attn_mm: an M5-class device; TS_METAL_QSA_ATTN_MM=0 keeps
    // the simdgroup kernel). The default window threshold is 8192 cells then, 12288 otherwise.
    bool qsa_attn_tensor_api(ggml_backend* backend);
    bool qsa_attn_encode(ggml_backend* backend, const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v,
                         const ggml_tensor* idx, const ggml_tensor* cnt, const ggml_tensor* mask_in, ggml_tensor* out,
                         int T, int n_head, int n_head_kv, int n_cells, int width, float scale);
    long long qsa_attn_count();

    // A prompt chunk's routed experts on Strata's grouped GEMM (tsm/kernels/tsm_moe_mm.metal, the Metal 4 tensor API):
    // the (token, expert) entries sorted by expert into 16-row tiles, gate+up+SwiGLU in one pass, down, then each
    // token's weighted sum in route order. Q2_0 gate/up/down stacks in one Metal buffer; TS_METAL_MOE_MM=0 turns it off.
    bool moe_mm_enabled();
    bool moe_mm_supported(ggml_backend* backend, int gate_type, int up_type, int down_type, int hidden, int n_ff,
                          int n_expert, int n_used);
    // x / out F32 [hidden, T]; ids I32 and wts F32 [n_used, T]; gate / up [hidden, n_ff, n_expert], down [n_ff, hidden,
    // n_expert]. wait: return only once the output is written. False with the last error set.
    bool moe_mm_encode(ggml_backend* backend, const ggml_tensor* x, const ggml_tensor* ids, const ggml_tensor* wts,
                       const ggml_tensor* gate, const ggml_tensor* up, const ggml_tensor* down, ggml_tensor* out,
                       int T, int hidden, int n_ff, int n_expert, int n_used, bool wait);
    long long moe_mm_count();
    // Drops moe_mm_encode's scratch (~0.8 GB at a 4096-row chunk of Flash Next): called once a prompt pass is done, so
    // decode does not carry it.
    void moe_mm_release();
    // Waits for the last command buffer a tsm encoder committed (ggml's synchronize does not cover them).
    void wait_last();

    // Copies srcs[i] into dsts[i] (the same byte counts, Metal buffers) in one blit pass on ggml-metal's queue, behind the
    // work already committed there, without waiting: the speculative state snapshot before a verify window, which a host
    // copy made wait for the accelerator and then memcpy ~115 MB (Flash Next, ~2 ms a step). Returns a retained handle
    // of its command buffer, or null with the last error set (the caller copies on the host then).
    void* copy_async(ggml_backend* backend, const ggml_tensor* const* srcs, const ggml_tensor* const* dsts, int n);
    // Waits for a copy_async handle's copies to complete; release_handle drops it (null is a no-op for both).
    void wait_handle(void* handle);
    void release_handle(void* handle);

    // ---- A prompt chunk's hyper-connection read in one pass on the matrix units (tsm/kernels/tsm_hc_read.metal, after
    // Strata's gr_norm_h / gemm_f16 / gr_epi_down / gemm_hc_up): xn = rms_norm(each stream) * gamma in fp16, the down
    // projection and the injection weights in one GEMM, silu(down / hc), and the up projection whose epilogue mixes
    // the streams, in place of ggml's norm, three matmuls and element-wise passes. The weights are dequantized to fp16
    // a chunk. TS_METAL_HC_READ=0 keeps the ggml path.
    struct HcReadIO
    {
        const ggml_tensor* res = nullptr;       // F32 [n_embd * hc, T], the residual (read only)
        const ggml_tensor* gamma = nullptr;     // F32 [n_embd * hc]
        const ggml_tensor* w_down = nullptr;    // [n_embd * hc, lr]
        const ggml_tensor* w_inject = nullptr;  // [n_embd * hc, hc]
        const ggml_tensor* w_up = nullptr;      // [lr, n_embd * hc]
        ggml_tensor* mixed = nullptr;           // F32 [n_embd, T], written
        ggml_tensor* inject = nullptr;          // F32 [hc, T], written
        int T = 0, n_embd = 0, hc = 0, lr = 0;
        float eps = 0.0f;
    };
    // Whether the kernels take these weight types (F32, F16, BF16, Q8_0) and shapes on this backend.
    bool hc_read_supported(ggml_backend* backend, int down_type, int inject_type, int up_type, int n_embd, int hc, int lr);
    bool hc_read_encode(ggml_backend* backend, const HcReadIO& io);
    long long hc_read_count();

    // ---- The end of a DFlash2 draft (tsm/kernels/tsm_df_tail.metal): the target's q5_K output head over the block's
    // proposal rows on the matrix units, the top-16 of each row, and the candidate selector's transition lattice over
    // q4_K codebooks of rank 256, in one command buffer behind the draft graph on ggml-metal's queue, in place of
    // ggml's per-row head, a vocabulary-wide top-k and the selector's small ops. The outputs are the ggml path's: the
    // candidates [gamma][16] and the lattice (16 anchor scores, then a [16 predecessor][16 candidate] matrix per later
    // position). TS_METAL_DF_TAIL=0 keeps the ggml path.
    struct DfTailIO
    {
        const ggml_tensor* h = nullptr;      // F32 [hidden, rows], the final normed rows; the proposals are rows 1..gamma
        const ggml_tensor* ph = nullptr;     // F32 [rank, gamma], the selector's hidden projection (scaled)
        const ggml_tensor* head = nullptr;   // q5_K [hidden, vocab]
        const ggml_tensor* pred = nullptr;   // q4_K [rank, vocab], the predecessor codebook
        const ggml_tensor* succ = nullptr;   // q4_K [rank, vocab], the successor codebook
        int gamma = 0, anchor = 0;
        float head_scale = 1.0f, pred_scale = 1.0f, succ_scale = 1.0f, logit_scale = 1.0f, softcap = 0.0f;
        int* cand_out = nullptr;             // host [gamma * 16]
        float* scores_out = nullptr;         // host [16 + 16 * 16 * (gamma - 1)]
        float* logits_out = nullptr;         // optional host [gamma * vocab] (the probe)
    };
    bool df_tail_enabled();
    bool df_tail_supported(ggml_backend* backend, int head_type, int hidden, int vocab, int rank, int top_k,
                           int pred_type, int succ_type, int gamma);
    // Encodes and commits the tail, waits for it, and copies the outputs to the host arrays.
    bool df_tail_run(ggml_backend* backend, const DfTailIO& io);
    long long df_tail_count();
}
