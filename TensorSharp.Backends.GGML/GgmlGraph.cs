// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    /// <summary>How a graph leaf's host bytes are bound (ggml_ops_graph_builder.cpp).</summary>
    public enum GgmlLeafRole
    {
        /// <summary>Read-only: the cached device copy keyed by the host pointer, a zero-copy wrap, or an upload.</summary>
        Weight = 0,
        /// <summary>Read-write state the graph updates in place (KV caches, recurrent states).</summary>
        State = 1,
        /// <summary>Per-run values supplied to <see cref="GgmlGraph.Run"/>.</summary>
        Input = 2,
    }

    /// <summary>Op codes of the generic graph (keep in sync with TsgGraphOp in ggml_ops_graph_builder.cpp).</summary>
    public enum GgmlGraphOp
    {
        GetRows = 1, Add = 2, Sub = 3, Mul = 4, Div = 5, Scale = 6, ScaleBias = 7,
        RmsNorm = 8, Norm = 9, L2Norm = 10, GroupNorm = 11, MulMat = 12, MulMatId = 13,
        Silu = 14, Gelu = 15, GeluQuick = 16, Sigmoid = 17, Softplus = 18, Exp = 19, Neg = 20,
        Tanh = 21, Relu = 22, Sqr = 23, Sqrt = 24, SwigluSplit = 25, GegluSplit = 26,
        Reshape = 27, View = 28, Permute = 29, Transpose = 30, Cont = 31, Cast = 32, Cpy = 33,
        SetRows = 34, RopeExt = 35, FlashAttnExt = 36, SoftMaxExt = 37, Argsort = 38, TopK = 39,
        SumRows = 40, SsmConv = 41, GatedDeltaNet = 42, Concat = 43, Clamp = 44, Repeat = 45,
        Pad = 46, Mean = 47, AddId = 48, Argmax = 49, KvWindow = 50, Swiglu = 51, Geglu = 52,
    }

    /// <summary>ggml tensor types used by graph leaves (ggml.h's enum ggml_type).</summary>
    public static class GgmlType
    {
        public const int F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8, I32 = 26, I64 = 27, BF16 = 30;
    }

    /// <summary>
    /// One ggml graph built node by node from managed code and run as a whole on the active
    /// device (ggml_ops_graph_builder.cpp). Node handles are small integers in emission order.
    /// After the first <see cref="Run"/> the graph is fixed; later runs replay it with new
    /// inputs. Not thread-safe; one graph is built and run by one thread at a time.
    /// </summary>
    public sealed class GgmlGraph : IDisposable
    {
        private IntPtr _handle;

        /// <param name="replayable">True for a graph run more than once (it then keeps its own
        /// compute memory); false borrows the backend's shared compute memory for one run.</param>
        public GgmlGraph(int maxNodes, bool replayable = false)
        {
            _handle = GgmlNative.TSGgml_GraphCreate(maxNodes, replayable ? 1 : 0);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("ggml graph creation failed: " + GgmlBasicOps.LastNativeError());
        }

        public bool IsBuilt { get; private set; }

        /// <summary>Nodes emitted so far.</summary>
        public int NodeCount { get; private set; }

        public unsafe int Leaf(GgmlLeafRole role, int type, ReadOnlySpan<long> ne, IntPtr data, long bytes)
        {
            fixed (long* pne = ne)
                return Check(GgmlNative.TSGgml_GraphLeaf(Handle, (int)role, type, ne.Length, pne, data, bytes), "leaf");
        }

        public int Op(GgmlGraphOp op, int a) => Op(op, stackalloc[] { a });

        public int Op(GgmlGraphOp op, int a, int b) => Op(op, stackalloc[] { a, b });

        public int Op(GgmlGraphOp op, int a, int b, int c) => Op(op, stackalloc[] { a, b, c });

        public unsafe int Op(GgmlGraphOp op, ReadOnlySpan<int> srcs, ReadOnlySpan<long> ip = default, ReadOnlySpan<float> fp = default)
        {
            fixed (int* ps = srcs)
            fixed (long* pi = ip)
            fixed (float* pf = fp)
                return Check(GgmlNative.TSGgml_GraphOp(Handle, (int)op, ps, srcs.Length, pi, ip.Length, pf, fp.Length), op.ToString());
        }

        /// <summary>ne0..ne3, nb0..nb3 and the ggml type of a node.</summary>
        public unsafe (long[] Ne, long[] Nb, int Type) Shape(int node)
        {
            long* ne = stackalloc long[4];
            long* nb = stackalloc long[4];
            int type;
            if (GgmlNative.TSGgml_GraphShape(Handle, node, ne, nb, &type) == 0)
                throw new InvalidOperationException("ggml graph shape query failed: " + GgmlBasicOps.LastNativeError());
            return (new[] { ne[0], ne[1], ne[2], ne[3] }, new[] { nb[0], nb[1], nb[2], nb[3] }, type);
        }

        /// <summary>Download <paramref name="node"/> into <paramref name="data"/> after every run.</summary>
        public void Output(int node, IntPtr data, long bytes)
        {
            if (GgmlNative.TSGgml_GraphOutput(Handle, node, data, bytes) == 0)
                throw new InvalidOperationException("ggml graph output failed: " + GgmlBasicOps.LastNativeError());
        }

        /// <summary>Run <paramref name="node"/> for its side effect (a cache or state write).</summary>
        public void Effect(int node)
        {
            if (GgmlNative.TSGgml_GraphEffect(Handle, node) == 0)
                throw new InvalidOperationException("ggml graph effect failed: " + GgmlBasicOps.LastNativeError());
        }

        /// <summary>Run the graph with this run's input values. False (with the native error) when it failed.</summary>
        public unsafe bool Run(ReadOnlySpan<int> inputs, ReadOnlySpan<IntPtr> data, ReadOnlySpan<long> bytes)
        {
            if (inputs.Length != data.Length || inputs.Length != bytes.Length)
                throw new ArgumentException("one data pointer and byte count per input");
            int ok;
            fixed (int* pi = inputs)
            fixed (IntPtr* pd = data)
            fixed (long* pb = bytes)
                ok = GgmlNative.TSGgml_GraphRun(Handle, pi, (void**)pd, pb, inputs.Length);
            IsBuilt = true;
            return ok != 0;
        }

        /// <summary>Whether a STATE leaf lives in device memory that outlives the run.</summary>
        public bool IsResident(int leaf) => GgmlNative.TSGgml_GraphLeafIsResident(Handle, leaf) != 0;

        private IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(GgmlGraph));

        private int Check(int node, string what)
        {
            if (node < 0)
                throw new InvalidOperationException($"ggml graph {what} failed: {GgmlBasicOps.LastNativeError()}");
            NodeCount = node + 1;
            return node;
        }

        public void Dispose()
        {
            if (_handle == IntPtr.Zero) return;
            GgmlNative.TSGgml_GraphFree(_handle);
            _handle = IntPtr.Zero;
        }
    }

    internal static partial class GgmlNative
    {
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial IntPtr TSGgml_GraphCreate(int maxNodes, int replayable);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial void TSGgml_GraphFree(IntPtr graph);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static unsafe partial int TSGgml_GraphLeaf(IntPtr graph, int role, int type, int nd, long* ne, IntPtr data, long bytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static unsafe partial int TSGgml_GraphOp(IntPtr graph, int op, int* srcs, int nsrc, long* ip, int nip, float* fp, int nfp);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static unsafe partial int TSGgml_GraphShape(IntPtr graph, int node, long* ne, long* nb, int* type);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial int TSGgml_GraphOutput(IntPtr graph, int node, IntPtr data, long bytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial int TSGgml_GraphEffect(IntPtr graph, int node);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static unsafe partial int TSGgml_GraphRun(IntPtr graph, int* inputs, void** data, long* bytes, int ninputs);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial int TSGgml_GraphLeafIsResident(IntPtr graph, int node);
    }
}
