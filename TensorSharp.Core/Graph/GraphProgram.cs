// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Graph
{
    /// <summary>
    /// The operations of a backend-neutral graph program. Each backend's executor plugin lowers
    /// them (see <see cref="IGraphExecutor"/>); the parameters follow ggml's semantics, which the
    /// models were written against (extents innermost first, byte strides, NeoX / multi-axis
    /// rotary modes).
    /// </summary>
    public enum GraphOp
    {
        GetRows = 1, Add = 2, Sub = 3, Mul = 4, Div = 5, Scale = 6, ScaleBias = 7,
        RmsNorm = 8, Norm = 9, L2Norm = 10, GroupNorm = 11, MulMat = 12, MulMatId = 13,
        Silu = 14, Gelu = 15, GeluQuick = 16, Sigmoid = 17, Softplus = 18, Exp = 19, Neg = 20,
        Tanh = 21, Relu = 22, Sqr = 23, Sqrt = 24, SwigluSplit = 25, GegluSplit = 26,
        Reshape = 27, View = 28, Permute = 29, Transpose = 30, Cont = 31, Cast = 32, Cpy = 33,
        SetRows = 34, RopeExt = 35, FlashAttnExt = 36, SoftMaxExt = 37, Argsort = 38, TopK = 39,
        SumRows = 40, SsmConv = 41, GatedDeltaNet = 42, Concat = 43, Clamp = 44, Repeat = 45,
        Pad = 46, Mean = 47, AddId = 48, Argmax = 49, KvWindow = 50, Swiglu = 51, Geglu = 52, RopeMulti = 53,
        /// <summary>Routed SwiGLU experts whose weights stay in host memory: srcs x [n, rows], ids (I32 [k, rows]),
        /// route weights ([k, rows]), then the stacked gate, up and down experts, and optionally the next offloaded
        /// layer's router and gate stack (a hint: an executor may read that layer's predicted experts ahead); Ip: the
        /// layer, n_ff. Result
        /// [n, rows] = sum over k of w * down(silu(gate x) * up x) for each row's experts. An executor with
        /// <see cref="GraphCapabilities.HostExperts"/> pauses the program there (an accelerator tier registered
        /// for the gate stack runs the routes it holds); the others do not take it.</summary>
        MoeHostExperts = 54,
        /// <summary>A lightning indexer's scores (ggml_lightning_indexer): srcs q [d, heads, rows], keys [d, 1, n],
        /// the heads' weights [heads, rows] and a mask [n, rows] (F16). Result [n, rows] = sum over heads of
        /// weight * relu(q_head . key) plus the mask.</summary>
        LightningIndexer = 55,
        /// <summary>A top-k selection's attention mask (ggml_top_k_mask): srcs mask [n_kv, rows], picks [n_top, rows]
        /// (I32), then, where Ip[0..2] are set, each pick's score [n_pick, rows] (a pick lives while its score is
        /// above -inf), the cells of each pick [n_cells, n_pick] (I32; else a pick is a cell) and cells every row keeps
        /// [n_extra, rows] (I32). Result [n_kv, rows]: the mask's value at a kept cell, -inf elsewhere; cells out of
        /// range are skipped.</summary>
        TopKMask = 56,
        /// <summary>This rank's partial [n, rows] summed over its tensor-parallel group (every rank gets the total): srcs the
        /// partial, which no other node reads. The ranks' programs run in lockstep (see <see cref="IGraphCollective"/>), so
        /// every rank reaches its reduce before any rank reads the total.</summary>
        AllReduce = 57,
        /// <summary>A 2-D convolution with zero padding (ggml_conv_2d): srcs the kernel [KW, KH, IC, OC] (F32 or F16) and x
        /// [W, H, IC, N] (F32). Ip: stride0, stride1, padding0, padding1, dilation0, dilation1 (axis 0 is the width). Result
        /// [OW, OH, OC, N] (F32), OW = (W + 2 p0 - d0 (KW - 1) - 1) / s0 + 1 (OH alike). With an F16 kernel x is read at F16
        /// (ggml's im2col columns for it), with an F32 kernel at F32.</summary>
        Conv2d = 58,
        /// <summary>Reflect padding along ne0 (ggml_pad_reflect_1d): srcs x (F32, contiguous). Ip: p0 (before), p1 (after),
        /// each less than ne0. Result [p0 + ne0 + p1, ne1, ne2, ne3]: column p0 + i is x's column i, column p0 - j (1 &lt;= j &lt;=
        /// p0) is x's column j and column p0 + ne0 - 1 + j (1 &lt;= j &lt;= p1) is x's column ne0 - 1 - j (the edge not
        /// repeated). Padding of two axes is two passes, the second over a transpose.</summary>
        PadReflect1d = 59,
        /// <summary>A depthwise 2-D convolution over rows of channels (ggml_conv_2d_dw_direct's arithmetic, in the layout the
        /// IR's token rows already have): srcs x [C, W, H, N] (channels innermost) and the kernel [KW, KH, 1, C] (ggml's
        /// depthwise kernel layout, F32). Ip: stride0, stride1, pad0, pad1, dilation0, dilation1 (index 0 along W, 1 along H;
        /// zero padding on both sides). Result [C, OW, OH, N] with OW = (W + 2 pad0 - dilation0 (KW - 1) - 1) / stride0 + 1
        /// (OH likewise): output (c, ow, oh, n) is the sum over the kernel taps, ky outer and kx inner, of
        /// kernel[kx, ky, 0, c] * x[c, ow stride0 + kx dilation0 - pad0, oh stride1 + ky dilation1 - pad1, n], taps that fall
        /// outside x contributing nothing.</summary>
        DepthwiseConv2d = 70,
        /// <summary>GELU in its exact form, 0.5 x (1 + erf(x / sqrt 2)) (ggml_gelu_erf; <see cref="Gelu"/> is ggml_gelu's tanh
        /// approximation). Executors without an erf kernel evaluate erf with Abramowitz and Stegun's 7.1.26 (|error| below
        /// 1.5e-7), as ggml-metal does.</summary>
        GeluErf = 71,
    }

    /// <summary>
    /// A tensor-parallel group's sum for executors whose ranks' programs are traced in lockstep on one thread: each
    /// rank's <see cref="GraphOp.AllReduce"/> hands its partial in, and the last rank's arrival sums them in place (the
    /// group's collective), before the trace emits any rank's next node.
    /// </summary>
    public interface IGraphCollective
    {
        int Degree { get; }
        /// <summary>Rank <paramref name="rank"/>'s partial (a reference the collective releases once it summed).</summary>
        void Contribute(int rank, Tensor partial);
    }

    /// <summary>How a leaf's bytes are bound.</summary>
    public enum GraphLeafRole
    {
        /// <summary>Read-only model bytes (a weight, a family constant), resident on the device.</summary>
        Weight = 0,
        /// <summary>Read-write bytes the program updates in place (KV caches, recurrent states).</summary>
        State = 1,
        /// <summary>Per-run values (activations, positions, masks, write indices).</summary>
        Input = 2,
    }

    /// <summary>Element formats of leaves: GGUF's tensor type ids (the formats weights arrive in).</summary>
    public static class GraphFormat
    {
        public const int F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8, I32 = 26, I64 = 27, BF16 = 30;
    }

    /// <summary>
    /// <see cref="GraphOp.FlashAttnExt"/>'s ip[2]: what its query rows are, as the model that emitted them says. The op's ip:
    /// [0] ggml's accumulation precision, [1] the most keys a mask row leaves finite (0: unknown), [2] one of these, and
    /// [3] a sliding window's band where the executor reads band masks on the device
    /// (<see cref="GraphCapabilities.BandMasksOnDevice"/>; such an op's [2] is <see cref="Unsaid"/>: its rows are decode rows or
    /// a verify window). An op that is neither marked nor banded carries at most two ip, so its program keeps the bytes and
    /// signature it had.
    /// </summary>
    public static class GraphAttentionRows
    {
        /// <summary>Not said (decode rows, a verify window, a drafter's rows): an executor keeps the arithmetic decode rows
        /// share.</summary>
        public const long Unsaid = 0;

        /// <summary>One sequence's prompt chunk: prompt tokens a plain and a speculating request forward through the same
        /// chunks, never a verify window's or a drafter's rows. An executor may attend them with arithmetic decode rows do not
        /// share (the CUDA executor's 8-bit prompt attention).</summary>
        public const long PromptChunk = 1;
    }

    /// <summary>A weight held in a quantized block layout (GGUF's formats): what a backend needs to
    /// run a product with it (its resident copy is found by <see cref="CacheKey"/>).</summary>
    public interface IGraphWeight
    {
        int Format { get; }
        /// <summary>Extents innermost first: input width, output rows, and (stacked experts) the expert count.</summary>
        long Ne0 { get; }
        long Ne1 { get; }
        long Ne2 { get; }
        /// <summary>The raw blocks (host memory; may be released once a backend made them resident).</summary>
        IntPtr Data { get; }
        /// <summary>The key a backend's resident copy is registered under.</summary>
        IntPtr CacheKey { get; }
        long Bytes { get; }
    }

    /// <summary>Stacked experts (<see cref="IGraphWeight.Ne2"/> of them) whose members a backend can
    /// also take one at a time (each with its own resident copy).</summary>
    public interface IGraphExpertWeights : IGraphWeight
    {
        IGraphWeight Expert(int index);
    }

    /// <summary>How a leaf spreads over tensor-parallel ranks: along <see cref="Axis"/>, <see cref="Ne"/>
    /// holding each (segment, rank)'s extent, segment s repeating <see cref="Nr"/>[s] times.</summary>
    public sealed record GraphSplit(int Axis, long[] Ne, uint[] Nr);

    public enum GraphNodeKind : byte { Leaf, Op, Effect, Output, Alias, Boundary, Fused, FusedEnd }

    /// <summary>
    /// What a fused region computes (see <see cref="GraphProgramBuilder.BeginFused"/>). The region's ops are
    /// the reference: an executor with a kernel for the kind may compute the region's results from its
    /// inputs in one dispatch; any other runs the ops.
    /// </summary>
    public enum GraphFusion
    {
        /// <summary>A gated delta rule's inputs for one sequence. Inputs: the conv window [convRows, channels]
        /// (time innermost), qkv [channels, rows], the conv kernel [width, channels], alpha and beta
        /// [vHeads, rows], dt and a [vHeads]. Ip: dk, kHeads, dv, vHeads. Fp: the L2 norm's eps. Results:
        /// the conv input [convRows + rows, channels] (window then rows, time innermost); q and k
        /// [dk, heads, rows, 1], each head L2-normed (x / sqrt(sum x^2 + eps)), of silu(ssm_conv); v
        /// [dv, vHeads, rows, 1] of silu(ssm_conv); g = softplus(alpha + dt) * a and beta = sigmoid(beta),
        /// both [1, vHeads, rows, 1].</summary>
        DeltaNetPrep = 1,
        /// <summary>rms_norm(y) * w * silu(z) per head (sigmoid(z) with Ip[0] = 1). Inputs: y and z [d, heads, rows],
        /// w [d]. Fp: eps. Result: [d, heads, rows].</summary>
        GatedRmsNorm = 2,
        /// <summary>A SwiGLU feed-forward: down · (silu(gate · x) * (up · x)). Inputs: x [n, rows], then the
        /// weights: gate_up (gate rows then up rows) and down, or (Ip[0] = 1) gate, up and down. With Ip[1] = 1 the
        /// projections have biases, which follow the weights in the same order (gate_up's [2 ff] and down's [n], or
        /// gate's, up's and down's): down · (silu(gate · x + b_gate) * (up · x + b_up)) + b_down. Result: [n, rows].</summary>
        SwigluFfn = 3,
        /// <summary>The routed experts of a softmax-gated MoE layer: the top-K experts of each row by its router
        /// logits, their softmax weights renormalized over the K (Ip[1] = 1) or as they are (Ip[1] = 0), and the
        /// weighted sum of their SwiGLU feed-forwards. Inputs: x [n, rows], the router logits [E, rows], then the
        /// stacked experts gate, up and down. Ip: K, renormalize. Result: [n, rows].</summary>
        RoutedExperts = 4,
        /// <summary>A gated delta-net layer's mixer over one sequence's prompt chunk, from its input projections to the
        /// gated output: the inputs' prep (holding a <see cref="DeltaNetPrep"/> region), the recurrence, the conv window
        /// and recurrent state updates (effects inside the region) and the gated norm (a <see cref="GatedRmsNorm"/>
        /// region). Inputs: qkv [channels, rows], z [dv * vHeads, rows], alpha and beta [vHeads, rows], the conv kernel
        /// [width, channels], dt and a [vHeads], the norm weight [dv], the conv window [width - 1, channels] and the
        /// recurrent state [dk, dv, vHeads, 1] (both written in place). Ip: dk, kHeads, dv, vHeads, the output gate
        /// (0 silu, 1 sigmoid), the conv width. Fp: eps. Result: the gated output [dv, vHeads, rows].</summary>
        DeltaNetPrompt = 5,
        /// <summary>A hyper-connection read: each of the hc residual streams RMS-normed and the row scaled by the gammas
        /// (xn), the low-rank gate sigmoid(up · silu(down · xn / hc)), the streams' gated mean and the injection
        /// weights inject · xn. Inputs: the streams [hc * n, rows], the gammas [hc * n], then the down [hc * n, lr],
        /// inject [hc * n, hc] and up [lr, hc * n] weights (no per-tensor scale). Ip: n, hc, lr. Fp: eps. Results: the
        /// mix [n, rows] and the injection weights [hc, rows].</summary>
        HcRead = 6,
        /// <summary>A prompt chunk's hyper-connection scatter: the residual streams plus the block output in every stream,
        /// weighted by 2 * sigmoid(inject / hc). Inputs: the residual streams [hc * n, rows], the block output [n, rows]
        /// and the read's injection weights [hc, rows]. Ip: n, hc. Result: the residual streams [hc * n, rows].</summary>
        HcScatter = 7,
        /// <summary>A prompt chunk's hyper-connection mix: the streams' mean of the normed rows gated by sigmoid of their
        /// gates. Inputs: the normed streams and the gates before their sigmoid, both [hc * n, rows]. Ip: n, hc. Result:
        /// the mix [n, rows].</summary>
        HcMix = 8,
        /// <summary>Ops that read only the region's inputs and whose results are read later (a shared expert beside the
        /// router and the routed experts): an executor may run them beside the step's other work (a second stream), and
        /// waits for them where their results are first read. Inputs: what the region reads. Results: what it computes.
        /// Executors that do not run it so compute the ops in place.</summary>
        ConcurrentBranch = 9,
        /// <summary>A softmax router's selection: each row's top-K experts by their softmax probabilities (a tie to the lower
        /// expert) and their weights - the probabilities, renormalized over the K (Ip[1] = 1) or as they are. Inputs: the
        /// router logits [E, rows]. Ip: K, renormalize. Results: the ids [K, rows] (I32) and the weights [1, K, rows].</summary>
        MoeSelect = 10,
        /// <summary>A gated attention output: attn * sigmoid(gate). Inputs: attn [d * heads, rows] and gate [d, heads, rows] (a
        /// view of the query projection, its heads interleaved with the queries). Result: [d * heads, rows].</summary>
        SigmoidGate = 11,
        /// <summary>A MoE layer's output plus its shared expert gated per row: moe + shared * sigmoid(x . w). Inputs: moe and
        /// shared [n, rows], x [n, rows] (the layer's input) and w [n] (ffn_gate_inp_shexp). Result: [n, rows].</summary>
        SharedExpertGate = 12,
        /// <summary>A sparse attention layer's block selection and its attention, for one sequence over its cache window (Qwen
        /// 3.8's QSA past its dense budget): the selection's ops (the pooled keys' gather, the lightning indexer, the top-k and
        /// the top-k mask) and the flash attention under that mask. Block b is cells b R .. b R + R - 1, visible to the row at
        /// KV index p iff b &lt; (p + 1) / R; its pooled key is the second half of its last member's row of the indexer cache.
        /// A row attends its <c>picks</c> best visible blocks whole (by score, a tie to the lower block; every visible block
        /// where there are no more) and its own incomplete tail block's cells up to p. Inputs: the rows' indexer queries
        /// [idxDim, idxHeads, rows] (rotated), the indexer cache [2 idxDim, capacity], the attention's queries [d, heads, rows],
        /// the K and V caches [d, capacity, kvHeads] and the rows' KV indices [rows] (the rows the step wrote). Ip: R, picks,
        /// the window (the keys the mask spans), idxDim, idxHeads, heads, kvHeads, d, the attention's key bound (its kvMax).
        /// Fp: the attention's scale, the indexer's weight of every head. Result: the attention [d, heads, rows].</summary>
        QsaAttention = 13,
        /// <summary>A sparse attention layer's indexer keys for one sequence's rows (Qwen 3.8's QSA): each row's raw key into
        /// its [raw | pooled] row of the indexer cache (the pooled half 0 times the raw key), and every block the rows complete
        /// pooled into its last member's row - the mean of its members' raw keys as the cache holds them, RMS-normed with the
        /// norm weight and rotated at its first cell's position. The writes are effects inside the region. The ops also pool
        /// again blocks complete before the step (a short chunk's last ones), to the values they hold; a kernel need not.
        /// Inputs: the rows' raw keys [d, rows], the indexer cache [2 d, capacity], the rows' KV indices [rows] (consecutive),
        /// the norm weight [d] and the rotary positions [n] of the first cells of the blocks the ops pool: from the first
        /// block the rows complete, or (Ip[2] = 1, a short chunk) the last n blocks complete once they are written. Ip: d, R, the
        /// short chunk flag, the rotary dims, mode and original context. Fp: the norm's eps, 1 / R, the rotary's base,
        /// frequency scale, extrapolation factor, attention factor, beta fast and beta slow. Result: the pooled write.</summary>
        QsaKeys = 14,
        /// <summary>A verify window's commit over the gated delta-net layers of one program (a pipeline stage's, a rank's):
        /// each layer's recurrent state after the window's first <c>kept</c> rows and its conv window after them. Inputs, five a
        /// layer: the state source, the state target, the window's saved scan inputs (q and k [saved rows, kHeads, S], v
        /// [saved rows, vHeads, S], the gates and the betas [saved rows, vHeads], each a region of its own), the conv history
        /// [convRows + capacity, channels] and the conv window [convRows, channels]. Ip: S (the key and value head width), vHeads,
        /// kHeads, the saved rows, convRows, channels, the history's pitch (floats a channel), kept, then three a layer: the rows
        /// to step, the source's and the target's offsets (floats). A layer with rows to step takes the gated delta rule over the
        /// first of them in its saved inputs from the source's state (on the decode rows' kernels: the window's rounding), the
        /// rows' outputs and the state after them written at the target's offset ([outputs | state]); with none, the source's
        /// state at the target's offset; with -1, its state stays. Every layer's conv window takes the history's columns
        /// [kept, kept + convRows) of each channel. The writes are effects and aliases inside the region. Result: the last
        /// layer's conv window write.</summary>
        DeltaNetCommit = 15,
        /// <summary>An attention layer's query and key prep and its cache writes for one sequence's rows: each query head of q
        /// and KV head of k RMS-normed with its weight (x * rsqrt(mean(x^2) + eps), then times the weight) and rotated at the row's
        /// position (ggml_rope_ext, single-axis); the rotated keys and the values written into the K and V caches at the rows' KV
        /// indices, rounded to the caches' type. The writes are effects inside the region. Inputs: q [d, heads, rows] and k
        /// [d, kvHeads, rows] (views of the projections, d contiguous: before the copies and norms the ops run), v
        /// [dv, kvHeads, rows], the q and k norm weights [d], the K and V caches [d, capacity, kvHeads] and [dv, capacity, kvHeads],
        /// the rows' rotary positions [rows] (I32) and their KV indices [rows]. Ip: heads, kvHeads, d, dv, the rotary dims, mode and
        /// original context, then (optional) 1 where only the attention reads the result, through its [d, rows, heads] permute (an
        /// executor may then lay it out head by head). Fp: the norms' eps, the rotary's base, frequency scale, extrapolation factor,
        /// attention factor, beta fast and beta slow. Result: the rotated queries [d, heads, rows].</summary>
        AttentionPrep = 16,
        /// <summary>DFlash2's grouped dynamic convolution of a block's rows: out[r][c] = sum over taps t &lt;= r of
        /// x[r - t][c] * base[side][t][c] + x[r - t][c] * taps[r][side * width * groups + t * groups + c / group], each product
        /// rounded on its own, each tap's two products added, and the taps added in order (tap 0's term, then the result plus
        /// each later tap's: zero on the rows before that tap reaches). Inputs: x [hidden, rows], taps [stride, rows] (the
        /// conv projection of the sublayer's input) and the base kernel [hidden, width, 2]. Ip: width, group, groups, side,
        /// stride, then (optional) a block's rows: the rows are blocks of that many one after another (several requests'
        /// blocks), each convolved on its own (r counts from its block's first row). Result: [hidden, rows].</summary>
        DFlashConv = 17,
        /// <summary>A diffusion transformer's adaLN modulation of a packed sequence: each row RMS-normed and times the norm
        /// weight, then n + n * scale + shift, where scale and shift are columns of the modulation matrix picked by the row's
        /// segment. Inputs: x [hidden, rows] (F32, rows contiguous), the norm weight [hidden], mods [hidden, columns] (F32,
        /// contiguous: column c is one modulation vector) and the segment table [3, segments] (I32: start, end, column base
        /// of each run of rows, sorted, contiguous and covering [0, rows); an input fixed for a request). Ip: the shift and
        /// scale offsets from a segment's column base. Fp: eps. Result: [hidden, rows]. The ops (<see
        /// cref="GraphDitRegions.AdaLnModulate"/>) norm the whole sequence once, then modulate and concatenate it segment
        /// by segment.</summary>
        AdaLnModulate = 18,
        /// <summary>A diffusion transformer's gated residual over a packed sequence: base + (delta + (mm + bias)) * gate,
        /// the gate a column of the modulation matrix picked by the row's segment. Inputs: base and mm [hidden, rows] (F32,
        /// rows contiguous; mm a projection's product), then the bias [hidden] (Ip[1] = 1) and the delta [hidden, rows] (Ip[2]
        /// = 1, a LoRA update the projection's kernel did not take), mods [hidden, columns] and the segment table [3,
        /// segments] (as <see cref="AdaLnModulate"/>'s). Ip: the gate's offset from a segment's column base, bias, delta.
        /// Result: [hidden, rows].</summary>
        GatedResidual = 19,
        /// <summary>Attention's q/k prep with a rotary table: each head of q and k RMS-normed (times its weight where Ip[1]
        /// says so), then x * cos + rotate_half(x) * sinSigned over its first rot dims, the rest passed through. The table
        /// holds a row's angles for the two halves of the rotated dims (any positions: fractional, several axes) and the
        /// sine's sign is on the table: sinSigned = [-sin(first half), sin(second half)], so the half-swapped head [x2, x1]
        /// times it is rotate_half(x) * sin. Inputs: q and k [D, heads, rows] (F32, D contiguous; views of a projection),
        /// then q's and k's norm weights [D] (as Ip[1]'s bits 0 and 1 say), cos and sinSigned [rot, rows], and with Ip[2] = 1
        /// v [D, heads, rows]. Ip: rot (even, at most D), norm weights (bit 0 q, bit 1 k), attend. Fp: eps, then with
        /// attend the attention's scale and V's power-of-two scale. Results: the rotated q and k [D, heads, rows]; with
        /// attend the region goes on through the unmasked attention that reads them (a nested <see cref="DitAttention"/>
        /// region), and its result is that attention [D * heads, rows]: softmax(scale q k^T) v per head with K and V at F16, V
        /// divided by its scale before and the result multiplied by it after (<see cref="GraphDitRegions.TableRopeQkNorm"/>).</summary>
        TableRopeQkNorm = 20,
        /// <summary>A projection with unmerged low-rank updates: w · x plus up_i · (down_i · x) on the rows of each part's slot.
        /// Inputs: x [in, rows] (F32), the weight w [in, out], then each part's down [in, rank] and up [rank, slot rows]
        /// (F16 or F32, the adapter's scale folded into up), in slot order. Ip: the slots that carry a part (bit s: rows
        /// [s * slot rows, (s + 1) * slot rows) of the output, at most three slots), the slot rows (0: out, one slot). Result
        /// [out, rows] (no bias).</summary>
        LoraLinear = 21,
        /// <summary>A diffusion transformer's unmasked attention over a packed sequence: softmax(scale q k^T) v per head, with K
        /// and V at F16, V divided by its power-of-two scale before (the F16 accumulator's headroom) and the result multiplied
        /// by it after. Inputs: q, k and v [D, heads, rows] (F32, D contiguous, any head and row strides: views of a
        /// projection). Fp: scale, V's scale. Result: [D * heads, rows]. The ops (<see cref="GraphDitRegions.Attention"/>) are
        /// the flash path of MiniMax-H3's native graph; the GGML executor lowers the region to that graph's own nodes
        /// (ggml_h3_attention.h: on ggml-metal's tensor API the keys padded to 64 and the queries chunked, which the ops cannot
        /// say), and <see cref="TableRopeQkNorm"/>'s attend form holds it.</summary>
        DitAttention = 22,
        /// <summary>A ViT's self-attention over equal-length sequences packed along the rows (a batch of image tiles), from
        /// the block's normed input: qkv = w · x (+ b), each head's q, k and v interleaved (head h's three at rows [3D·h,
        /// 3D·(h+1)) of a qkv row); then per sequence q and k RMS-normed per head without a gain and rotated over their first
        /// rot dims by the tables [rot, seq] every sequence shares, and the unmasked attention per head (a nested <see
        /// cref="TableRopeQkNorm"/> attend form); the sequences' results concatenated. Inputs: x [dim, rows] (F32, rows =
        /// sequences · seq), w [dim, 3·dim], the bias [3·dim] where Ip[3] says, cos and sinSigned [rot, seq]. Ip: heads, rot,
        /// sequences, bias. Fp: eps, the attention's scale, V's power-of-two scale. Result: [dim, rows]. The ops (<see
        /// cref="GraphDitRegions.VitAttention"/>) are the unfused graph of MiniMax-H3's ViT decoder (h3_vit_decode); ggml_cuda
        /// serves it as that decoder's fused nodes: the product on cuBLAS F16 and one cuDNN attention over the whole batch,
        /// with the bias added in its prep.</summary>
        VitAttention = 23,
        /// <summary>A residual branch closed by its last projection, bias and per-channel layer scale: base + (W · act(x) + b)
        /// * scale. Ip[0] is the form: 0 a projection (inputs base [hidden, rows], x [in, rows], w [in, hidden], then b
        /// [hidden] and scale [hidden] as Ip[1] says; act is the identity); 1 a SwiGLU FFN (inputs base, x, the gate|up
        /// weight w1 [in, 2·ffn], its bias b1 [2·ffn], the down weight w2 [ffn, hidden], its bias b2 [hidden], then scale:
        /// act(x) = swiglu(w1 · x + b1), W = w2, b = b2). Ip[1]: bit 0 the biases, bit 1 the scale. Result: [hidden, rows].
        /// The ops (<see cref="GraphDitRegions.ScaledResidual"/>) are MiniMax-H3's ViT decoder's unfused residuals; ggml_cuda
        /// serves the F16 form as its fused ones (the products on cuBLAS F16, the SwiGLU and b1 in the down product's input,
        /// the bias and scale in H3_FUSED_SCALED_RESIDUAL).</summary>
        ScaledResidual = 24,
        /// <summary>Several sequences' verify-window recurrences of one gated delta-net layer (a batched verify: each sequence on
        /// its own state): the sequences' GatedDeltaNet ops, each the window form (its rows' outputs alone, its scan inputs
        /// saved, the state read and left). Inputs: per sequence its op's q, k, v, g, beta, state and saved-inputs leaf, in
        /// sequence order. Ip: the sequences, then each one's saved-input rows. Results: each sequence's op result, in order.
        /// An executor may run every sequence's scan in one dispatch; each sequence's values are its op's.</summary>
        DeltaNetScans = 25,
        /// <summary>Several block drafter lanes' attentions (several requests' blocks drafted in one program, each lane's rows
        /// over its own KV ring under its own mask): the lanes' FlashAttention ops and the concatenation of their results along
        /// the rows. Inputs: per lane its op's q, K, V and mask, in lane order. Ip: the lanes. Fp: the scale. Result: the lanes'
        /// attentions [d, heads, lanes * rows]. An executor may run every lane's in one dispatch; each lane's values are its op's.</summary>
        RingAttentionLanes = 26,
        /// <summary>Several sequences' attentions of one layer (a batched verify's windows, each over its own caches): the
        /// sequences' cache writes and attention ops, which read the region's inputs and each sequence's own caches alone.
        /// Inputs: the step's q, k and v. Results: each sequence's attention [heads * dv, its rows], in order. An executor may run
        /// the region's ops at its end and launch the sequences' attention kernels together (each sequence's values its ops');
        /// any other computes the ops in place.</summary>
        SequenceAttentions = 27,
        /// <summary>Several sequences' delta-rule inputs of one gated delta-net layer (a batched verify's windows, each sequence
        /// with its own conv window and history): each sequence's <see cref="DeltaNetPrep"/> region and the copy of its conv
        /// input into its conv history. Inputs: per sequence its conv window, qkv, alpha and beta (its prep region's), its conv
        /// history's view [convRows + rows, channels] and, with ip[5] set, its window's saved scan inputs (the leaf its
        /// GatedDeltaNet window op saves q | k | v | g | beta into), in sequence order; then the conv kernel, dt and a. Ip: dk,
        /// k heads, dv, v heads, the sequences, the saved-inputs flag. Fp: eps. Results: per sequence its prep's q, k, v, g and
        /// beta, in sequence order. An executor may run every sequence's prep in one dispatch, writing each conv input into its
        /// history (and its results into its saved inputs' planes, read there by the scans); each value is its op's.</summary>
        DeltaNetPreps = 28,
        /// <summary>Several sequences' gated norms of one gated delta-net layer (a batched verify's windows) and the concatenation
        /// of their results along the rows: each sequence's <see cref="GatedRmsNorm"/> region, then the concatenation. Inputs:
        /// per sequence its norm's y and z, in sequence order; then the norm weight. Ip: the sigmoid flag, the sequences. Fp:
        /// eps. Result: the concatenation [heads * d, every sequence's rows]. An executor may run every sequence's norm in one
        /// dispatch into the concatenation; each value is its op's.</summary>
        GatedRmsNorms = 29,
    }

    /// <summary>One recorded entry of a program. Leaves and ops are nodes (numbered in order);
    /// effects, outputs and aliases refer to them.</summary>
    public readonly struct GraphNode
    {
        public GraphNodeKind Kind { get; init; }
        public GraphOp Op { get; init; }
        public int[] Srcs { get; init; }
        public long[] Ip { get; init; }
        public float[] Fp { get; init; }
        public GraphLeafRole Role { get; init; }
        public int Format { get; init; }
        public long[] Ne { get; init; }
        /// <summary>A leaf's bytes (its identity for Weight and State; null for Input).</summary>
        public IntPtr Data { get; init; }
        public long Bytes { get; init; }
        /// <summary>The node an effect, output or alias refers to.</summary>
        public int Node { get; init; }
        public GraphSplit? Split { get; init; }
        /// <summary>A leaf's object for backends that run on tensors: the <see cref="Tensor"/> of a weight
        /// or state, or the <see cref="IGraphWeight"/> of a quantized weight (null for inputs).</summary>
        public object? Handle { get; init; }
    }

    /// <summary>A recorded program: what one step computes, independent of any backend.</summary>
    public sealed class GraphProgram
    {
        internal GraphProgram(GraphNode[] entries, int nodeCount, long[] outputFloats, bool replayable, bool reorder,
            int[]? tensorParallelRanks, (ulong, ulong) signature, bool wideTokenProducts = false)
        {
            Entries = entries;
            NodeCount = nodeCount;
            OutputFloats = outputFloats;
            Replayable = replayable;
            Reorder = reorder;
            TensorParallelRanks = tensorParallelRanks;
            Signature = signature;
            WideTokenProducts = wideTokenProducts;
        }

        public IReadOnlyList<GraphNode> Entries { get; }
        public int NodeCount { get; }
        /// <summary>F32 values of each output, in output order.</summary>
        public long[] OutputFloats { get; }
        /// <summary>The program runs again with new inputs (its plan keeps its own memory).</summary>
        public bool Replayable { get; }
        /// <summary>The backend may reorder independent work for concurrency.</summary>
        public bool Reorder { get; }
        /// <summary>Ranks the program spreads over (tensor parallelism), or null.</summary>
        public int[]? TensorParallelRanks { get; }
        /// <summary>A hash of everything that shapes the program (not its per-run input values).</summary>
        public (ulong, ulong) Signature { get; }
        /// <summary>The program's products have thousands of rows (a diffusion denoiser's step over a whole image): an
        /// executor may run its weight products on wide-GEMM routes it keeps for that, serve regions it fuses only at that
        /// width, and drop a power-of-two scale pair around a product such a route takes, which keeps F32's exponent range
        /// (<see cref="GraphProgramBuilder.MarkWideTokenProducts"/>). The GGML executor's on ggml_cuda: cuBLAS BF16 or int8
        /// products, cuDNN attention, and the diffusion-transformer regions as its fused nodes; elsewhere it changes
        /// nothing.</summary>
        public bool WideTokenProducts { get; }
    }

    /// <summary>
    /// Records a <see cref="GraphProgram"/>. In hash-only mode nothing is kept but the signature, so
    /// a caller can look up a compiled plan for a step before recording it.
    /// </summary>
    public sealed class GraphProgramBuilder
    {
        private readonly List<GraphNode>? _entries;
        private readonly List<long> _outputs = new();
        private ulong _h1 = 14695981039346656037UL, _h2 = 0x9E3779B97F4A7C15UL;
        private ulong _s1 = 14695981039346656037UL, _s2 = 0x9E3779B97F4A7C15UL;
        private bool _signatureOnly;
        private bool _wideTokenProducts;

        public GraphProgramBuilder(bool record) => _entries = record ? new List<GraphNode>() : null;

        /// <summary>The program will carry <see cref="GraphProgram.WideTokenProducts"/>. Part of the signature (also of a hash-only
        /// trace, so a plan looked up by it is a program that has the attribute); before any entry, so an executor that
        /// lowers as entries arrive knows it from the start.</summary>
        public void MarkWideTokenProducts()
        {
            if (NodeCount != 0 || _outputs.Count != 0)
                throw new InvalidOperationException("WideTokenProducts is a program attribute: mark it before the first entry.");
            if (_wideTokenProducts) return;
            _wideTokenProducts = true;
            Mix(9);
        }

        /// <summary>Whether <see cref="MarkWideTokenProducts"/> was called.</summary>
        public bool WideTokenProducts => _wideTokenProducts;

        public bool Recording => _entries != null;

        /// <summary>Also hands every entry to this sink as it is recorded (a streaming executor).</summary>
        public IGraphSink? Sink { get; set; }

        private void Emit(int node, in GraphNode entry)
        {
            _entries?.Add(entry);
            Sink?.Entry(node, entry);
        }

        private bool Keeping => _entries != null || Sink != null;
        public int NodeCount { get; private set; }
        public int OutputCount => _outputs.Count;
        public (ulong, ulong) Signature => (_h1, _h2);

        /// <summary>The signature without the state leaves' identities: two steps that run the same kernels over different
        /// caches and states (another slot's) share it.</summary>
        public (ulong, ulong) StructuralSignature => (_s1, _s2);

        public void Mix(long value)
        {
            ulong v = unchecked((ulong)value);
            _h1 = (_h1 ^ v) * 1099511628211UL;
            _h2 = (_h2 ^ (v + 0x9E3779B97F4A7C15UL + (_h2 << 6) + (_h2 >> 2))) * 0xC2B2AE3D27D4EB4FUL;
            if (_signatureOnly) return;
            _s1 = (_s1 ^ v) * 1099511628211UL;
            _s2 = (_s2 ^ (v + 0x9E3779B97F4A7C15UL + (_s2 << 6) + (_s2 >> 2))) * 0xC2B2AE3D27D4EB4FUL;
        }

        public void Mix(ReadOnlySpan<long> values) { Mix(values.Length); foreach (long v in values) Mix(v); }

        public void Mix(ReadOnlySpan<int> values) { Mix(values.Length); foreach (int v in values) Mix(v); }

        public void Mix(ReadOnlySpan<float> values) { Mix(values.Length); foreach (float v in values) Mix(BitConverter.SingleToInt32Bits(v)); }

        public int Leaf(GraphLeafRole role, int format, ReadOnlySpan<long> ne, IntPtr data, long bytes, GraphSplit? split = null, object? handle = null)
        {
            Mix(1); Mix((int)role); Mix(format); Mix(ne);
            if (role != GraphLeafRole.Input)
            {
                _signatureOnly = role == GraphLeafRole.State;
                Mix(data.ToInt64());
                _signatureOnly = false;
                Mix(bytes);
            }
            if (split != null) { Mix(100 + split.Axis); Mix(split.Ne); }
            if (Keeping) Emit(NodeCount, new GraphNode { Kind = GraphNodeKind.Leaf, Role = role, Format = format, Ne = ne.ToArray(), Data = data, Bytes = bytes, Split = split, Handle = handle });
            return NodeCount++;
        }

        public int Op(GraphOp op, int a) => Op(op, stackalloc[] { a });

        public int Op(GraphOp op, int a, int b) => Op(op, stackalloc[] { a, b });

        public int Op(GraphOp op, ReadOnlySpan<int> srcs, ReadOnlySpan<long> ip = default, ReadOnlySpan<float> fp = default)
        {
            foreach (int src in srcs)
                if (src < 0 || src >= NodeCount) throw new ArgumentOutOfRangeException(nameof(srcs), $"{op}: source node {src} does not exist");
            Mix(2); Mix((int)op); Mix(srcs); Mix(ip); Mix(fp);
            if (Keeping) Emit(NodeCount, new GraphNode { Kind = GraphNodeKind.Op, Op = op, Srcs = srcs.ToArray(), Ip = ip.ToArray(), Fp = fp.ToArray() });
            return NodeCount++;
        }

        /// <summary>Run <paramref name="node"/> for its side effect (a cache or state write).</summary>
        public void Effect(int node)
        {
            Mix(3); Mix(node);
            if (Keeping) Emit(-1, new GraphNode { Kind = GraphNodeKind.Effect, Node = node });
        }

        /// <summary>Write <paramref name="node"/>'s result in place inside the state leaf
        /// <paramref name="baseLeaf"/> at <paramref name="offset"/> bytes; a backend that cannot runs the
        /// <paramref name="fallback"/> copy instead.</summary>
        public void Alias(int node, int baseLeaf, long offset, int fallback)
        {
            Mix(5); Mix(node); Mix(baseLeaf); Mix(offset); Mix(fallback);
            if (Keeping) Emit(-1, new GraphNode { Kind = GraphNodeKind.Alias, Node = node, Srcs = new[] { baseLeaf, fallback }, Bytes = offset });
        }

        /// <summary>A scheduling hint, not a computation: <paramref name="node"/> closes a unit of work (a
        /// layer). An executor that builds work lazily may start running what precedes it there; one
        /// that compiles the whole program ignores it.</summary>
        public void Boundary(int node)
        {
            Mix(6); Mix(node);
            if (Keeping) Emit(-1, new GraphNode { Kind = GraphNodeKind.Boundary, Node = node });
        }

        /// <summary>
        /// Open a region of <paramref name="kind"/> over <paramref name="inputs"/> (nodes that exist now). The ops
        /// up to <see cref="EndFused"/> compute its results from the inputs alone; an executor with a kernel for the
        /// kind may run that instead, and nodes inside are then computed only if something else reads them. Writes
        /// inside it (effects, aliases) are the region's where its kind says so: the kernel makes them, else they run
        /// in order when the region closes.
        /// </summary>
        public void BeginFused(GraphFusion kind, ReadOnlySpan<int> inputs, ReadOnlySpan<long> ip = default, ReadOnlySpan<float> fp = default)
        {
            foreach (int src in inputs)
                if (src < 0 || src >= NodeCount) throw new ArgumentOutOfRangeException(nameof(inputs), $"{kind}: input node {src} does not exist");
            Mix(7); Mix((int)kind); Mix(inputs); Mix(ip); Mix(fp);
            if (Keeping) Emit(-1, new GraphNode { Kind = GraphNodeKind.Fused, Node = (int)kind, Srcs = inputs.ToArray(), Ip = ip.ToArray(), Fp = fp.ToArray() });
        }

        /// <summary>Close the open region; <paramref name="results"/> are its results in the kind's order.</summary>
        public void EndFused(ReadOnlySpan<int> results)
        {
            Mix(8); Mix(results);
            if (Keeping) Emit(-1, new GraphNode { Kind = GraphNodeKind.FusedEnd, Srcs = results.ToArray() });
        }

        /// <summary>A result of <paramref name="floats"/> F32 values; returns its output index.</summary>
        public int Output(int node, long floats)
        {
            Mix(4); Mix(node); Mix(floats);
            if (Keeping) Emit(-1, new GraphNode { Kind = GraphNodeKind.Output, Node = node, Bytes = floats });
            _outputs.Add(floats);
            return _outputs.Count - 1;
        }

        public GraphProgram ToProgram(bool replayable, bool reorder, int[]? tensorParallelRanks = null)
        {
            if (_entries == null) throw new InvalidOperationException("A hash-only trace has no program.");
            return new GraphProgram(_entries.ToArray(), NodeCount, _outputs.ToArray(), replayable, reorder, tensorParallelRanks, Signature,
                _wideTokenProducts);
        }
    }
}
