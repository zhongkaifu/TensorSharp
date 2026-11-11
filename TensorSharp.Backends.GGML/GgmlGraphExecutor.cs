// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using TensorSharp.Graph;

namespace TensorSharp.GGML
{
    /// <summary>
    /// The GGML backends' graph executor plugin: a program becomes one ggml graph (the generic native
    /// builder), weights bind to their cached device copies, states update in place, and a replayable
    /// program keeps its graph and memory. Tensor-parallel programs run on ggml's meta device.
    /// </summary>
    public sealed class GgmlGraphExecutor : IGraphExecutor
    {
        private static readonly Dictionary<GgmlBackendType, GgmlGraphExecutor> s_executors = new();

        [ModuleInitializer]
        internal static void Register()
            => GraphExecutors.Register(typeof(GgmlAllocator), allocator => For(((GgmlAllocator)allocator).Context.BackendType));

        public static GgmlGraphExecutor For(GgmlBackendType backend)
        {
            lock (s_executors)
            {
                if (!s_executors.TryGetValue(backend, out GgmlGraphExecutor executor))
                    s_executors[backend] = executor = new GgmlGraphExecutor(backend);
                return executor;
            }
        }

        private GgmlGraphExecutor(GgmlBackendType backend)
        {
            Backend = backend;
            bool metal = backend == GgmlBackendType.Metal;
            Capabilities = new GraphCapabilities
            {
                // Metal and CUDA read row-contiguous strided views (the fused Qwen graphs rely on it).
                StridedViews = metal || backend == GgmlBackendType.Cuda,
                // ggml-metal finishes reading a head's recurrent state before writing it.
                InPlaceStateAlias = metal,
                // Metal's concurrency reorder (+8% decode on Qwen 3.5 A3B).
                ConcurrencyReorder = metal,
                DecodeWindowRows = metal ? 64 : 256,
                // ggml-metal's tensor-API flash attention takes F16 K/V in windows of 64 only: a 1.8K prompt's
                // attention took 110 ms over F32 fresh rows of exact length, 51 ms over the padded cache window (M5 Pro).
                PaddedCacheAttention = metal,
                TensorParallel = backend == GgmlBackendType.Cuda,
                BindsHostPointers = true,
                SeparateDeviceCopies = backend is GgmlBackendType.Cuda or GgmlBackendType.Vulkan,
                QuantizedKvCache = true,
                // Routed experts in host memory run at a pause in the graph (ggml_ops_graph_builder.cpp).
                HostExperts = true,
            };
        }

        public GgmlBackendType Backend { get; }

        public GraphCapabilities Capabilities { get; }

        public IGraphPlan Compile(GraphProgram program)
        {
            ArgumentNullException.ThrowIfNull(program);
            int maxNodes = Math.Max(1024, program.NodeCount + 64);
            GgmlGraph g = program.TensorParallelRanks != null
                ? new GgmlGraph(maxNodes, program.Replayable, program.TensorParallelRanks)
                : new GgmlGraph(maxNodes, program.Replayable, program.Reorder && Capabilities.ConcurrencyReorder);
            var outputs = new List<float[]>();
            var pins = new List<GCHandle>();
            try
            {
                foreach (GraphNode n in program.Entries)
                {
                    switch (n.Kind)
                    {
                        case GraphNodeKind.Leaf:
                            int leaf = g.Leaf((GgmlLeafRole)(int)n.Role, n.Format, n.Ne, n.Data, n.Bytes);
                            if (n.Split != null) g.Split(leaf, n.Split.Axis, n.Split.Ne, n.Split.Nr);
                            break;
                        case GraphNodeKind.Op: g.Op((GgmlGraphOp)(int)n.Op, n.Srcs, n.Ip, n.Fp); break;
                        case GraphNodeKind.Effect: g.Effect(n.Node); break;
                        case GraphNodeKind.Alias: g.Alias(n.Node, n.Srcs[0], n.Bytes, n.Srcs[1]); break;
                        // A region the backend has a kernel for (Metal's prompt-width delta-net layers) runs as one
                        // device operation between graph segments; any other builds its ops.
                        case GraphNodeKind.Fused: g.FusedBegin(n.Node, n.Srcs, n.Ip, n.Fp); break;
                        case GraphNodeKind.FusedEnd: g.FusedEnd(n.Srcs); break;
                        case GraphNodeKind.Output:
                            var values = new float[n.Bytes];
                            GCHandle pin = GCHandle.Alloc(values, GCHandleType.Pinned);
                            outputs.Add(values);
                            pins.Add(pin);
                            g.Output(n.Node, pin.AddrOfPinnedObject(), n.Bytes * sizeof(float));
                            break;
                    }
                }
                return new Plan(g, outputs.ToArray(), pins.ToArray());
            }
            catch
            {
                g.Dispose();
                foreach (GCHandle pin in pins) pin.Free();
                throw;
            }
        }

        public void SyncToHost(IntPtr data, long bytes) => GgmlBasicOps.SyncHostBuffer(data, bytes);

        public void Invalidate(IntPtr data) => GgmlBasicOps.InvalidateHostBuffer(data);

        private sealed class Plan : IGraphPlan
        {
            private readonly GgmlGraph _graph;
            private readonly GCHandle[] _pins;

            public Plan(GgmlGraph graph, float[][] outputs, GCHandle[] pins)
            {
                _graph = graph;
                Outputs = outputs;
                _pins = pins;
            }

            public float[][] Outputs { get; }

            public string LastError { get; private set; }

            public bool Run(ReadOnlySpan<int> inputs, ReadOnlySpan<IntPtr> data, ReadOnlySpan<long> bytes)
            {
                if (_graph.Run(inputs, data, bytes)) return true;
                LastError = GgmlBasicOps.LastNativeError();
                return false;
            }

            public bool IsLeafResident(int node) => _graph.IsResident(node);

            public void Dispose()
            {
                _graph.Dispose();
                foreach (GCHandle pin in _pins) if (pin.IsAllocated) pin.Free();
            }
        }
    }
}
