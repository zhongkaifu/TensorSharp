// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.Graph;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// Emits one <see cref="GraphProgram"/> as a trace: weights, per-run inputs, ops and fusion regions, with the
    /// program's signature over everything that shapes it (ops, parameters, extents and the bound weights). The
    /// per-run inputs' values ride each run and are not part of the signature.
    ///
    /// This is the part of program emission that does not depend on what is emitted: the shared decoder runtime's
    /// step (<c>StandardDecoderModel.GraphStep</c>, which adds slots, caches, positions and advancing) and the
    /// diffusion runtime's denoiser and VAE programs both build on it, so the two runtimes record programs the same
    /// way and every executor plugin runs either.
    /// </summary>
    public class GraphEmitter : IDisposable
    {
        private readonly IGraphExecutor _executor;
        private protected readonly GraphProgramBuilder _b;
        private readonly List<GCHandle> _pins = new();
        private protected readonly List<int> _inputs = new();
        private protected readonly List<IntPtr> _inputData = new();
        private protected readonly List<long> _inputBytes = new();
        private protected readonly List<object> _inputHandles = new();
        private readonly HashSet<int> _strided = new();   // nodes whose layout is not contiguous

        /// <summary>A trace for <paramref name="executor"/>: it says how leaves bind (host bytes or objects) and
        /// compiles the program. <paramref name="record"/> false keeps only the signature (a plan lookup).</summary>
        public GraphEmitter(IGraphExecutor executor, bool record = true) : this(record)
        {
            _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        }

        /// <summary>A trace whose executor a derived emitter supplies (<see cref="Executor"/>).</summary>
        private protected GraphEmitter(bool record) => _b = new GraphProgramBuilder(record);

        /// <summary>The executor the program is emitted for, read where a leaf's binding depends on it.</summary>
        protected virtual IGraphExecutor Executor => _executor;

        /// <summary>A hash of everything that shapes the program (not its per-run input values).</summary>
        public (ulong, ulong) Signature => _b.Signature;

        /// <summary>The signature without the state leaves' identities: two programs that run the same kernels over
        /// different states share it.</summary>
        public (ulong, ulong) StructuralSignature => _b.StructuralSignature;

        /// <summary>Whether the trace records a program (false: only its signature).</summary>
        public bool Recording => _b.Recording;

        /// <summary>The program <see cref="ToProgram"/> (or a derived emitter's build) recorded last.</summary>
        public GraphProgram Program { get; private protected set; }

        /// <summary>A streaming executor's sink: entries and input values go there as they are emitted.</summary>
        public IGraphSink Sink { get => _b.Sink; init => _b.Sink = value; }

        /// <summary>Mix a value that shapes the program but no leaf or op carries into its signature.</summary>
        public void Mix(long value) => _b.Mix(value);

        /// <summary>The builder the trace records on, for helpers that emit onto a builder (the diffusion-transformer
        /// regions, <see cref="GraphDitRegions"/>); what they record is part of this trace.</summary>
        public GraphProgramBuilder Builder => _b;

        // ---- ops

        public int Op(GraphOp op, int a) => Op(op, stackalloc[] { a });
        public int Op(GraphOp op, int a, int b) => Op(op, stackalloc[] { a, b });
        public int Op(GraphOp op, int a, int b, int c) => Op(op, stackalloc[] { a, b, c });

        public int Op(GraphOp op, ReadOnlySpan<int> srcs, ReadOnlySpan<long> ip = default, ReadOnlySpan<float> fp = default)
            => _b.Op(op, srcs, ip, fp);

        /// <summary>Run <paramref name="node"/> for its side effect (a cache or state write).</summary>
        public void Effect(int node) => _b.Effect(node);

        /// <summary>A layer's output: a point where a lazily building executor may start running.</summary>
        public void Boundary(int node) => _b.Boundary(node);

        /// <summary>The ops up to <see cref="EndFused"/> are a region of <paramref name="kind"/> (see
        /// <see cref="GraphFusion"/>): an executor with its kernel computes the results in one dispatch.</summary>
        public void BeginFused(GraphFusion kind, ReadOnlySpan<int> inputs, ReadOnlySpan<long> ip = default, ReadOnlySpan<float> fp = default)
            => _b.BeginFused(kind, inputs, ip, fp);

        public void EndFused(ReadOnlySpan<int> results) => _b.EndFused(results);

        /// <summary>Write <paramref name="node"/>'s result in place inside the state leaf
        /// <paramref name="baseLeaf"/> at <paramref name="offset"/> bytes; where the base is not
        /// resident, the <paramref name="fallback"/> copy runs instead (see GgmlGraph.Alias).</summary>
        public void Alias(int node, int baseLeaf, long offset, int fallback) => _b.Alias(node, baseLeaf, offset, fallback);

        /// <summary>A program result: <paramref name="floats"/> F32 values of <paramref name="node"/>;
        /// returns its index among the program's results.</summary>
        public int Output(int node, long floats) => _b.Output(node, floats);

        // ---- leaves

        private protected int Leaf(GraphLeafRole role, int type, ReadOnlySpan<long> ne, IntPtr data, long bytes, GraphSplit split = null, object handle = null)
            => _b.Leaf(role, type, ne, data, bytes, split, handle);

        /// <summary>A weight held in a block layout (<see cref="IGraphWeight"/>): bound by its device-cache key.</summary>
        public int Weight(IGraphWeight weight)
        {
            ArgumentNullException.ThrowIfNull(weight);
            IntPtr key = weight.CacheKey != IntPtr.Zero ? weight.CacheKey : weight.Data;
            ReadOnlySpan<long> ne = weight.Ne2 > 1 ? stackalloc long[] { weight.Ne0, weight.Ne1, weight.Ne2 } : stackalloc long[] { weight.Ne0, weight.Ne1 };
            return Leaf(GraphLeafRole.Weight, weight.Format, ne, key, weight.Bytes, null, weight);
        }

        /// <summary>Read-only model bytes of <paramref name="format"/> at a host address (a mapped file's tensor, a
        /// family constant), with the given extents. An executor that binds host pointers keeps its device copy under
        /// <paramref name="data"/>; <paramref name="handle"/> is what an executor binding objects reads.</summary>
        public int HostWeight(int format, ReadOnlySpan<long> ne, IntPtr data, long bytes, object handle = null)
            => Leaf(GraphLeafRole.Weight, format, ne, data, bytes, null, handle);

        /// <summary>An F32 tensor (device constants, norms) as a read-only leaf.</summary>
        public virtual int TensorWeight(Tensor t, params long[] ne)
            => TensorLeaf(GraphLeafRole.Weight, t, GraphFormat.F32, ne.Length == 0 ? Extents(t) : ne);

        private protected int TensorLeaf(GraphLeafRole role, Tensor t, int type, ReadOnlySpan<long> ne, GraphSplit split = null)
            => Leaf(role, type, ne, LeafIdentity(t), t.Storage.ByteLength, split, t);

        /// <summary>A tensor leaf's bytes for an executor that binds host memory, else a stable identity
        /// (a device storage's host pointer would sync it every step).</summary>
        private protected IntPtr LeafIdentity(Tensor t) => Executor.Capabilities.BindsHostPointers
            ? TensorComputePrimitives.GetStoragePointer(t)
            : GraphIdentity.Of(t.Storage, t.StorageOffset);

        /// <summary>A tensor's extents innermost first (ggml order).</summary>
        private protected static long[] Extents(Tensor t)
        {
            var ne = new long[t.DimensionCount];
            for (int i = 0; i < ne.Length; i++) ne[i] = t.Sizes[t.DimensionCount - 1 - i];
            return ne;
        }

        // ---- per-run inputs

        /// <summary>An F32 activation [cols, rows] read from a host tensor every run.</summary>
        public int Input(Tensor t, long cols, long rows)
        {
            int node = Leaf(GraphLeafRole.Input, GraphFormat.F32, stackalloc long[] { cols, rows }, IntPtr.Zero, 0);
            IntPtr data = Executor.Capabilities.BindsHostPointers ? TensorComputePrimitives.GetStoragePointer(t) : IntPtr.Zero;
            RecordInput(node, data, cols * rows * sizeof(float), t, null);
            return node;
        }

        /// <summary>I32 values [count] read from <paramref name="values"/> every run.</summary>
        public int InputI32(int[] values, long count)
        {
            int node = Leaf(GraphLeafRole.Input, GraphFormat.I32, stackalloc long[] { count }, IntPtr.Zero, 0);
            RecordInput(node, Pin(values), count * sizeof(int), null, values);
            return node;
        }

        /// <summary>I32 values [<paramref name="ne0"/>, <paramref name="ne1"/>] read from <paramref name="values"/> every run (a
        /// segment table: <see cref="GraphDitRegions.SegmentTable"/>).</summary>
        public int InputI32(int[] values, long ne0, long ne1)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (values.LongLength != ne0 * ne1) throw new ArgumentException($"[{ne0}, {ne1}] I32 values need {ne0 * ne1} ints, not {values.LongLength}");
            int node = Leaf(GraphLeafRole.Input, GraphFormat.I32, stackalloc long[] { ne0, ne1 }, IntPtr.Zero, 0);
            RecordInput(node, Pin(values), values.LongLength * sizeof(int), null, values);
            return node;
        }

        /// <summary>F32 values [ne...] read from <paramref name="values"/> every run.</summary>
        public int InputF32(float[] values, params long[] ne)
        {
            int node = Leaf(GraphLeafRole.Input, GraphFormat.F32, ne, IntPtr.Zero, 0);
            RecordInput(node, Pin(values), (long)values.Length * sizeof(float), null, values);
            return node;
        }

        /// <summary>An F16 input [ne...] read from <paramref name="values"/> every run (a family's mask).</summary>
        public int InputF16(ushort[] values, params long[] ne)
        {
            int node = Leaf(GraphLeafRole.Input, GraphFormat.F16, ne, IntPtr.Zero, 0);
            RecordInput(node, Pin(values), values.LongLength * sizeof(ushort), null, values);
            return node;
        }

        /// <summary>Keep an input's run binding. A derived emitter that tracks how its inputs change between runs
        /// records the input's kind here; <paramref name="values"/> is the bound array, if any.</summary>
        protected virtual void RecordInput(int node, IntPtr data, long bytes, object handle, Array values)
            => AppendInput(node, data, bytes, handle);

        /// <summary>The binding every input keeps: its node, host bytes, byte count and handle, and the sink's copy.</summary>
        private protected void AppendInput(int node, IntPtr data, long bytes, object handle)
        {
            _b.Sink?.InputData(node, data, bytes, handle);
            _inputs.Add(node);
            _inputData.Add(data);
            _inputBytes.Add(bytes);
            _inputHandles.Add(handle);
        }

        /// <summary>Pin <paramref name="values"/> for the trace's lifetime (its runs read them in place).</summary>
        private protected IntPtr Pin(Array values)
        {
            GCHandle handle = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(handle);
            return handle.AddrOfPinnedObject();
        }

        // ---- layout

        public int Reshape(int x, params long[] ne) => Op(GraphOp.Reshape, stackalloc[] { x }, ne);

        /// <summary>A view of an F32 node; recorded as strided unless its rows are packed.</summary>
        public int View2(int x, long ne0, long ne1, long nb1, long offset)
        {
            int node = Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 2, ne0, ne1, 1, 1, nb1, 0, 0, offset });
            if (ne1 > 1 && nb1 != ne0 * 4) _strided.Add(node);
            return node;
        }

        public int View3(int x, long ne0, long ne1, long ne2, long nb1, long nb2, long offset)
        {
            int node = Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 3, ne0, ne1, ne2, 1, nb1, nb2, 0, offset });
            if ((ne1 > 1 && nb1 != ne0 * 4) || (ne2 > 1 && nb2 != ne0 * ne1 * 4)) _strided.Add(node);
            return node;
        }

        public int View4(int x, long ne0, long ne1, long ne2, long ne3, long nb1, long nb2, long nb3, long offset)
        {
            int node = Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 4, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset });
            if ((ne1 > 1 && nb1 != ne0 * 4) || (ne2 > 1 && nb2 != ne0 * ne1 * 4) || (ne3 > 1 && nb3 != ne0 * ne1 * ne2 * 4))
                _strided.Add(node);
            return node;
        }

        /// <summary>Permute axes (strided).</summary>
        public int Permute(int x, long a0, long a1, long a2, long a3)
        {
            int node = Op(GraphOp.Permute, stackalloc[] { x }, stackalloc long[] { a0, a1, a2, a3 });
            _strided.Add(node);
            return node;
        }

        public int Transpose(int x)
        {
            int node = Op(GraphOp.Transpose, x);
            _strided.Add(node);
            return node;
        }

        /// <summary>A contiguous copy of <paramref name="x"/>, or <paramref name="x"/> itself when it
        /// already is contiguous (every op result is; views, permutes and transposes may not be).</summary>
        public int Cont(int x) => _strided.Contains(x) ? Op(GraphOp.Cont, x) : x;

        /// <summary>The backend's norms, unary ops and recurrent kernels read row-contiguous strided
        /// views (ggml-metal, ggml-cuda); elsewhere a strided input is copied contiguous first.</summary>
        public bool StridedInputs => Executor.Capabilities.StridedViews;

        // ---- composites

        public int RmsNorm(int x, int weight, float eps)
            => Op(GraphOp.Mul, Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps }), weight);

        /// <summary>rms_norm(x, eps / n) / sqrt(n) over ne0: the L2 normalization ggml's GDN path uses.</summary>
        public int L2Norm(int x, long n, float eps)
            => Op(GraphOp.Scale, stackalloc[] { Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps / n }) },
                default, stackalloc[] { 1.0f / MathF.Sqrt(n) });

        /// <summary>Flash attention of q [d, rows, heads] over k/v [d, kv, kv_heads]: [dv, heads, rows]. <paramref name="mask"/>
        /// -1 attends every key. <paramref name="kvMax"/> &gt; 0 bounds the finite entries of every mask row, which lets a
        /// backend attend only those (ggml_flash_attn_ext_set_n_kv_max). <paramref name="promptChunk"/>: the rows are one
        /// sequence's prompt chunk (<see cref="GraphAttentionRows.PromptChunk"/>).</summary>
        /// <param name="band">A sliding-window layer's window (ip[3]; ip[2] is then the rows' kind, see
        /// <see cref="GraphAttentionRows"/>) where the executor reads such masks on the device
        /// (<see cref="GraphCapabilities.BandMasksOnDevice"/>), 0 elsewhere.</param>
        public int FlashAttention(int q, int k, int v, int mask, float scale, int kvMax = 0, bool promptChunk = false, int band = 0)
        {
            Span<int> srcs = stackalloc[] { q, k, v, mask };
            // ip[2] only on a prompt chunk's op or a device band's (GraphAttentionRows), ip[3] only on a device band's: every
            // other op keeps the ip, bytes and signature it had.
            long rows = promptChunk ? GraphAttentionRows.PromptChunk : GraphAttentionRows.Unsaid;
            ReadOnlySpan<long> ip = band > 0 ? stackalloc long[] { 10 /* F32 accumulation */, Math.Max(kvMax, 0), rows, band }
                : promptChunk ? stackalloc long[] { 10, Math.Max(kvMax, 0), rows }
                : kvMax > 0 ? stackalloc long[] { 10, kvMax } : stackalloc long[] { 10 };
            ObserveAttention(ip, band);
            return Op(GraphOp.FlashAttnExt, mask >= 0 ? srcs : srcs[..3], ip, stackalloc[] { scale, 0f, 0f });
        }

        /// <summary>Called with every attention op's ip (and its device band, 0 for none) before it is emitted.</summary>
        protected virtual void ObserveAttention(ReadOnlySpan<long> ip, int band) { }

        // ---- programs and runs

        /// <summary>Marks the program as a step whose products have thousands of rows, a diffusion denoiser's step over a
        /// whole image (<see cref="GraphProgramBuilder.MarkWideTokenProducts"/>). Call it before the first entry.</summary>
        public void MarkWideTokenProducts() => _b.MarkWideTokenProducts();

        /// <summary>The recorded program (see <see cref="GraphProgramBuilder.ToProgram"/>).</summary>
        public GraphProgram ToProgram(bool replayable, bool reorder = false, int[] tensorParallelRanks = null)
        {
            if (!_b.Recording) throw new InvalidOperationException("a hash-only trace records no program");
            return Program = _b.ToProgram(replayable, reorder, tensorParallelRanks);
        }

        /// <summary>Compile the recorded program with <see cref="Executor"/>.</summary>
        public IGraphPlan Compile(bool replayable, bool reorder = false) => Executor.Compile(ToProgram(replayable, reorder));

        /// <summary>This trace's per-run inputs, for a run of several plans together.</summary>
        public GraphRunInputs RunInputs() => new(_inputs.ToArray(), _inputData.ToArray(), _inputBytes.ToArray());

        /// <summary>Run <paramref name="plan"/> (compiled from this trace or one of the same signature) on this trace's inputs.</summary>
        public virtual bool Run(IGraphPlan plan) => plan.Run(CollectionsMarshal.AsSpan(_inputs), CollectionsMarshal.AsSpan(_inputData),
            CollectionsMarshal.AsSpan(_inputBytes), CollectionsMarshal.AsSpan(_inputHandles));

        public void Dispose()
        {
            foreach (GCHandle pin in _pins) pin.Free();
            _pins.Clear();
        }
    }
}
