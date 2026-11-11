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
        /// quantized weights: gate_up (gate rows then up rows) and down, or (Ip[0] = 1) gate, up and down.
        /// Result: [n, rows].</summary>
        SwigluFfn = 3,
        /// <summary>The routed experts of a softmax-gated MoE layer: the top-K experts of each row by its router
        /// logits, their softmax weights renormalized over the K (Ip[1] = 1) or as they are (Ip[1] = 0), and the
        /// weighted sum of their SwiGLU feed-forwards. Inputs: x [n, rows], the router logits [E, rows], then the
        /// stacked experts gate, up and down. Ip: K, renormalize. Result: [n, rows].</summary>
        RoutedExperts = 4,
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
            int[]? tensorParallelRanks, (ulong, ulong) signature)
        {
            Entries = entries;
            NodeCount = nodeCount;
            OutputFloats = outputFloats;
            Replayable = replayable;
            Reorder = reorder;
            TensorParallelRanks = tensorParallelRanks;
            Signature = signature;
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

        public GraphProgramBuilder(bool record) => _entries = record ? new List<GraphNode>() : null;

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

        public void Mix(long value)
        {
            ulong v = unchecked((ulong)value);
            _h1 = (_h1 ^ v) * 1099511628211UL;
            _h2 = (_h2 ^ (v + 0x9E3779B97F4A7C15UL + (_h2 << 6) + (_h2 >> 2))) * 0xC2B2AE3D27D4EB4FUL;
        }

        public void Mix(ReadOnlySpan<long> values) { Mix(values.Length); foreach (long v in values) Mix(v); }

        public void Mix(ReadOnlySpan<int> values) { Mix(values.Length); foreach (int v in values) Mix(v); }

        public void Mix(ReadOnlySpan<float> values) { Mix(values.Length); foreach (float v in values) Mix(BitConverter.SingleToInt32Bits(v)); }

        public int Leaf(GraphLeafRole role, int format, ReadOnlySpan<long> ne, IntPtr data, long bytes, GraphSplit? split = null, object? handle = null)
        {
            Mix(1); Mix((int)role); Mix(format); Mix(ne);
            if (role != GraphLeafRole.Input) { Mix(data.ToInt64()); Mix(bytes); }
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
        /// up to <see cref="EndFused"/> compute its results from the inputs alone (no effects, no state); an
        /// executor with a kernel for the kind may run that instead, and nodes inside are then computed
        /// only if something else reads them.
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
            return new GraphProgram(_entries.ToArray(), NodeCount, _outputs.ToArray(), replayable, reorder, tensorParallelRanks, Signature);
        }
    }
}
