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
    /// The emission side of a <see cref="NetworkGraph"/> run: a network's call written as graph IR nodes (no native calls),
    /// with the shapes tracked so the layer helpers (NetworkBuilder.Layers.cs) can lay out views, permutes and products. The
    /// trace's signature covers everything that shapes the program (ops, parameters, extents, the bound weights and states);
    /// per-run inputs (activations, positions, masks, indices) ride the run. A builder lives for one
    /// <see cref="NetworkGraph.Run(Action{NetworkBuilder})"/>: emit with it inside the callback only.
    /// </summary>
    public sealed partial class NetworkBuilder : IDisposable
    {
        private readonly NetworkGraph _graph;
        private readonly GraphProgramBuilder _b;
        private readonly Dictionary<string, NetValue> _weightLeaves = new(StringComparer.Ordinal);
        private readonly Dictionary<Tensor, NetValue> _stateLeaves = new(ReferenceEqualityComparer.Instance);
        private readonly List<GCHandle> _pins = new();
        private readonly List<int> _inputs = new();
        private readonly List<IntPtr> _inputData = new();
        private readonly List<long> _inputBytes = new();
        private readonly List<object> _inputHandles = new();
        private readonly bool _bindsHost;

        internal NetworkBuilder(NetworkGraph graph, bool record)
        {
            _graph = graph;
            _b = new GraphProgramBuilder(record);
            _bindsHost = graph.Executor.Capabilities.BindsHostPointers;
        }

        /// <summary>The weights the network reads.</summary>
        public NetworkWeights Weights => _graph.Weights;

        /// <summary>What the executor that runs the program does (strided reads, replays, ...).</summary>
        public GraphCapabilities Capabilities => _graph.Executor.Capabilities;

        /// <summary>Outputs declared so far.</summary>
        public int OutputCount => _b.OutputCount;

        /// <summary>Nodes emitted so far.</summary>
        public int NodeCount => _b.NodeCount;

        internal (ulong, ulong) Signature => _b.Signature;

        // ================================================================ raw ops

        /// <summary>
        /// Emit <paramref name="op"/> over <paramref name="srcs"/> with its ggml parameters (see <see cref="GraphOp"/>); the
        /// caller states the result's extents (innermost first) and whether it is a strided view. The escape hatch for ops the
        /// helpers do not wrap.
        /// </summary>
        public NetValue Op(GraphOp op, ReadOnlySpan<NetValue> srcs, ReadOnlySpan<long> resultNe, ReadOnlySpan<long> ip = default,
            ReadOnlySpan<float> fp = default, bool strided = false, int format = GraphFormat.F32)
        {
            Span<int> ids = srcs.Length <= 16 ? stackalloc int[srcs.Length] : new int[srcs.Length];
            for (int i = 0; i < srcs.Length; i++)
            {
                if (!srcs[i].Valid) throw new ArgumentException($"{op}: source {i} is not a value");
                ids[i] = srcs[i].Id;
            }
            int id = _b.Op(op, ids, ip, fp);
            return Value(id, resultNe, strided, format);
        }

        private NetValue Emit(GraphOp op, NetValue a, long ne0, long ne1, long ne2, long ne3, ReadOnlySpan<long> ip = default,
            ReadOnlySpan<float> fp = default, bool strided = false)
            => new(_b.Op(op, stackalloc[] { a.Id }, ip, fp), ne0, ne1, ne2, ne3, strided, GraphFormat.F32);

        private NetValue Emit(GraphOp op, NetValue a, NetValue b, long ne0, long ne1, long ne2, long ne3, ReadOnlySpan<long> ip = default,
            ReadOnlySpan<float> fp = default, bool strided = false)
            => new(_b.Op(op, stackalloc[] { a.Id, b.Id }, ip, fp), ne0, ne1, ne2, ne3, strided, GraphFormat.F32);

        private static NetValue Value(int id, ReadOnlySpan<long> ne, bool strided, int format)
        {
            if (ne.Length < 1 || ne.Length > 4) throw new ArgumentException("a value has 1 to 4 extents");
            return new NetValue(id, ne[0], ne.Length > 1 ? ne[1] : 1, ne.Length > 2 ? ne[2] : 1, ne.Length > 3 ? ne[3] : 1, strided, format);
        }

        private static long[] Extents(ReadOnlySpan<long> ne)
        {
            if (ne.Length < 1 || ne.Length > 4) throw new ArgumentException("1 to 4 extents");
            return ne.ToArray();
        }

        // ================================================================ weights and states

        /// <summary>Weight <paramref name="name"/> as loaded: a product weight [row width, rows], a dense one with its GGUF
        /// extents.</summary>
        public NetValue Weight(string name)
        {
            if (_weightLeaves.TryGetValue(name, out NetValue v)) return v;
            NetworkWeight w = Weights.Get(name);
            v = w.Quantized != null ? QuantizedLeaf(w.Quantized) : DenseLeaf(w.Dense, w.Ne);
            _weightLeaves[name] = v;
            return v;
        }

        /// <summary>Weight <paramref name="name"/> read with extents <paramref name="ne"/> (the same values): a vector of
        /// [1, 1, C] as [C], a [1, 1, Cin, Cout] pointwise kernel as [Cin, Cout], a 4-D patch kernel as a matrix. A product
        /// weight is re-read as rows of ne0 (<see cref="NetworkWeights.Reshaped"/>).</summary>
        public NetValue Weight(string name, params long[] ne)
        {
            NetworkWeight w = Weights.Get(name);
            long n = 1;
            foreach (long d in ne) n *= d;
            if (n != w.Elements) throw new ArgumentException($"weight '{name}' has {w.Elements} values, not [{string.Join(", ", ne)}]");
            if (w.Quantized != null)
            {
                if (ne.Length > 2 && ne[2] * (ne.Length > 3 ? ne[3] : 1) != 1)
                    throw new ArgumentException($"weight '{name}' is a product weight: read it as [row width, rows]");
                return Weight(Weights.Reshaped(name, ne[0]).Name);
            }
            string key = $"{name}|{string.Join("x", ne)}";
            if (_weightLeaves.TryGetValue(key, out NetValue v)) return v;
            v = DenseLeaf(w.Dense, ne);
            _weightLeaves[key] = v;
            return v;
        }

        /// <summary>Rows [<paramref name="firstRow"/>, + <paramref name="rows"/>) of product weight <paramref name="name"/>
        /// (<see cref="NetworkWeights.RowBlock"/>): a product with it computes those rows only.</summary>
        public NetValue WeightRows(string name, long firstRow, long rows) => Weight(Weights.RowBlock(name, firstRow, rows).Name);

        /// <summary>The per-tensor multiplier of a product weight (an NVFP4 sidecar; 1 otherwise).</summary>
        public float WeightScale(string name) => Weights.Get(name).Quantized?.Scale ?? 1f;

        private NetValue QuantizedLeaf(QuantizedWeight q)
        {
            // GGML binds the host bytes (its device copies are keyed on them); the tensor executors read the weight object.
            int id = _b.Leaf(GraphLeafRole.Weight, q.GgmlType, stackalloc long[] { q.Ne0, q.Ne1 }, q.Data, q.RawBytes, null, q);
            return new NetValue(id, q.Ne0, q.Ne1, 1, 1, false, q.GgmlType);
        }

        private NetValue DenseLeaf(Tensor t, ReadOnlySpan<long> ne)
        {
            int id = _b.Leaf(GraphLeafRole.Weight, GraphFormat.F32, ne, LeafIdentity(t), t.Storage.ByteLength - t.StorageOffset * sizeof(float), null, t);
            return Value(id, ne, false, GraphFormat.F32);
        }

        /// <summary>A read-write tensor on the network's allocator that programs update in place (a streaming encoder's carried
        /// columns, a codec's K/V tail): read it as a node, write it with <see cref="Write"/>. Its identity is part of the
        /// program's signature.</summary>
        public NetValue State(Tensor t, params long[] ne)
        {
            ArgumentNullException.ThrowIfNull(t);
            if (_stateLeaves.TryGetValue(t, out NetValue v)) return v;
            int format = t.ElementType switch
            {
                DType.Float32 => GraphFormat.F32,
                DType.Float16 => GraphFormat.F16,
                DType.Int32 => GraphFormat.I32,
                _ => throw new NotSupportedException($"a state of {t.ElementType}"),
            };
            if (ne == null || ne.Length == 0) ne = ExtentsOf(t);
            int id = _b.Leaf(GraphLeafRole.State, format, ne, LeafIdentity(t), t.Storage.ByteLength - t.StorageOffset * t.ElementType.Size(), null, t);
            v = Value(id, ne, false, format);
            _stateLeaves[t] = v;
            return v;
        }

        /// <summary>Copy <paramref name="value"/> into <paramref name="destination"/> (a state or a view of one) when the program
        /// runs (the write is an effect: it runs whether or not anything reads it).</summary>
        public void Write(NetValue value, NetValue destination)
        {
            if (value.Elements != destination.Elements)
                throw new ArgumentException($"write of {value} into {destination}: element counts differ");
            NetValue copy = Emit(GraphOp.Cpy, value, destination, destination.Ne0, destination.Ne1, destination.Ne2, destination.Ne3);
            _b.Effect(copy.Id);
        }

        private IntPtr LeafIdentity(Tensor t) => _bindsHost ? TensorComputePrimitives.GetStoragePointer(t) : GraphIdentity.Of(t.Storage, t.StorageOffset);

        private static long[] ExtentsOf(Tensor t)
        {
            var ne = new long[t.DimensionCount];
            for (int i = 0; i < ne.Length; i++) ne[i] = t.Sizes[t.DimensionCount - 1 - i];
            return ne;
        }

        // ================================================================ per-run inputs

        /// <summary>F32 values with extents <paramref name="ne"/> (rows of features: [features, rows]), read every run.</summary>
        public NetValue Input(float[] values, params long[] ne)
        {
            ArgumentNullException.ThrowIfNull(values);
            CheckCount(values.LongLength, ne);
            int id = _b.Leaf(GraphLeafRole.Input, GraphFormat.F32, ne, IntPtr.Zero, 0);
            AddInput(id, Pin(values), values.LongLength * sizeof(float), null);
            return Value(id, ne, false, GraphFormat.F32);
        }

        /// <summary>An F32 tensor as an input with extents <paramref name="ne"/> (default: its own): a device tensor another
        /// program left (<see cref="NetworkGraph.RunToTensors"/>), read where it lies by executors that bind tensors.</summary>
        public NetValue Input(Tensor t, params long[] ne)
        {
            ArgumentNullException.ThrowIfNull(t);
            if (t.ElementType != DType.Float32) throw new NotSupportedException($"a tensor input of {t.ElementType}");
            if (ne == null || ne.Length == 0) ne = ExtentsOf(t);
            CheckCount(t.ElementCount(), ne);
            if (!t.IsContiguous()) throw new ArgumentException("a tensor input is contiguous");
            int id = _b.Leaf(GraphLeafRole.Input, GraphFormat.F32, ne, IntPtr.Zero, 0);
            AddInput(id, _bindsHost ? TensorComputePrimitives.GetStoragePointer(t) : IntPtr.Zero, t.ElementCount() * sizeof(float), t);
            return Value(id, ne, false, GraphFormat.F32);
        }

        /// <summary>I32 values (positions, row indices) with extents <paramref name="ne"/> (default [count]).</summary>
        public NetValue InputI32(int[] values, params long[] ne)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (ne == null || ne.Length == 0) ne = new long[] { values.LongLength };
            CheckCount(values.LongLength, ne);
            int id = _b.Leaf(GraphLeafRole.Input, GraphFormat.I32, ne, IntPtr.Zero, 0);
            AddInput(id, Pin(values), values.LongLength * sizeof(int), null);
            return Value(id, ne, false, GraphFormat.I32);
        }

        /// <summary>F16 values given as their bits (an attention mask: 0 or 0xFC00 = -inf) with extents <paramref name="ne"/>.</summary>
        public NetValue InputF16(ushort[] bits, params long[] ne)
        {
            ArgumentNullException.ThrowIfNull(bits);
            CheckCount(bits.LongLength, ne);
            int id = _b.Leaf(GraphLeafRole.Input, GraphFormat.F16, ne, IntPtr.Zero, 0);
            AddInput(id, Pin(bits), bits.LongLength * sizeof(ushort), null);
            return Value(id, ne, false, GraphFormat.F16);
        }

        /// <summary>F32 zeros with extents <paramref name="ne"/> (a cold stream's state, a zero border).</summary>
        public NetValue Zeros(params long[] ne)
        {
            long n = 1;
            foreach (long d in ne) n *= d;
            return Input(new float[n], ne);
        }

        private static void CheckCount(long count, ReadOnlySpan<long> ne)
        {
            if (ne.Length < 1 || ne.Length > 4) throw new ArgumentException("an input has 1 to 4 extents");
            long n = 1;
            foreach (long d in ne) n *= d;
            if (n != count) throw new ArgumentException($"{count} values for extents [{string.Join(", ", ne.ToArray())}]");
        }

        private IntPtr Pin(Array values)
        {
            GCHandle handle = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(handle);
            return handle.AddrOfPinnedObject();
        }

        private void AddInput(int node, IntPtr data, long bytes, object handle)
        {
            _inputs.Add(node);
            _inputData.Add(data);
            _inputBytes.Add(bytes);
            _inputHandles.Add(handle);
        }

        // ================================================================ outputs

        /// <summary>Return <paramref name="x"/> from the run as F32 values (packed first where it is a strided view); returns
        /// its index in <see cref="NetworkGraph.Run(Action{NetworkBuilder})"/>'s results.</summary>
        public int Output(NetValue x)
        {
            if (x.Format != GraphFormat.F32) throw new NotSupportedException($"an output of format {x.Format} (outputs are F32)");
            NetValue packed = Cont(x);
            return _b.Output(packed.Id, packed.Elements);
        }

        // ================================================================ run

        internal GraphProgram ToProgram(bool replayable, bool reorder) => _b.ToProgram(replayable, reorder);

        internal bool Run(IGraphPlan plan) => plan.Run(CollectionsMarshal.AsSpan(_inputs), CollectionsMarshal.AsSpan(_inputData),
            CollectionsMarshal.AsSpan(_inputBytes), CollectionsMarshal.AsSpan(_inputHandles));

        public void Dispose()
        {
            foreach (GCHandle pin in _pins) pin.Free();
            _pins.Clear();
        }
    }
}
