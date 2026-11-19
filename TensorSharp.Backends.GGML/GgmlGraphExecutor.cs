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
    public sealed class GgmlGraphExecutor : IRankedGraphExecutor
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
                // ggml-cuda groups a KV head's queries only over K whose length is a multiple of 256: over exact-length
                // rows Qwen3.6-35B-A3B's 3.6K prompt spent 133.6 ms in flash attention, Qwen35Model's padded window 31.5
                // (A40, nsys).
                PaddedCacheAttention = metal || backend == GgmlBackendType.Cuda,
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
            int served = 0, lowered = 0;
            try
            {
                if (program.WideTokenProducts) g.MarkWideTokenProducts();
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
                        // device operation between graph segments, a diffusion-transformer region of a wide program on
                        // CUDA as the DiT offload's nodes inside the graph; any other builds its ops.
                        case GraphNodeKind.Fused:
                            if (g.FusedBegin(n.Node, n.Srcs, n.Ip, n.Fp, out bool low)) served++;
                            if (low) lowered++;
                            break;
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
                return new Plan(g, outputs.ToArray(), pins.ToArray(), served, lowered);
            }
            catch
            {
                g.Dispose();
                foreach (GCHandle pin in pins) pin.Free();
                throw;
            }
        }

        /// <summary>Per-rank graphs need ranks with a device collective: without one every segment boundary is a host
        /// round trip (tsg::tp_fused_available).</summary>
        public bool RunsRanks(int ranks) => Backend == GgmlBackendType.Cuda && ranks > 1 && GgmlBasicOps.TensorParallelFusedAvailable(ranks);

        /// <summary>A graph is built for the active rank's device: compile with rank <paramref name="rank"/> active.</summary>
        public IGraphPlan CompileForRank(GraphProgram program, int rank)
        {
            int previous = GgmlBasicOps.GetActiveRank();
            GgmlBasicOps.SetActiveRank(rank);
            try { return Compile(program); }
            finally { GgmlBasicOps.SetActiveRank(previous); }
        }

        /// <summary>The ranks' graphs segment by segment, each all-reduce partial summed between segments
        /// (ggml_ops_graph_builder.cpp's TSGgml_GraphRunRanked over tsg::tp_execute_plans). An exact sum of a partial wider
        /// than the transport's F32 limit runs in pieces of 32,764 floats (a 2048-row prompt chunk's: ~256 collectives).</summary>
        public bool RunRanked(IGraphPlan[] plans, GraphRunInputs[] inputs, bool exactSums, out string error)
        {
            ArgumentNullException.ThrowIfNull(plans);
            ArgumentNullException.ThrowIfNull(inputs);
            var graphs = new GgmlGraph[plans.Length];
            for (int r = 0; r < plans.Length; r++)
                graphs[r] = plans[r] is Plan p ? p.Graph : throw new ArgumentException("every plan must be this executor's", nameof(plans));
            bool ok = GgmlGraph.RunRanked(graphs, Array.ConvertAll(inputs, i => i.Nodes), Array.ConvertAll(inputs, i => i.Data),
                Array.ConvertAll(inputs, i => i.Bytes), exactSums);
            error = ok ? null : GgmlBasicOps.LastNativeError();
            return ok;
        }

        /// <summary>How many of <paramref name="plan"/>'s fused regions a device kernel computes instead of their ops (served
        /// when compiled; a region nested in a served one is part of it).</summary>
        public static int ServedRegions(IGraphPlan plan)
            => plan is Plan p ? p.ServedRegions : throw new ArgumentException("not a ggml graph plan", nameof(plan));

        /// <summary>How many of <paramref name="plan"/>'s fused regions the backend built as graph nodes of its own instead of
        /// their ops (GraphFusion.DitAttention: MiniMax-H3's attention nodes; a region nested in a served one is part of it).</summary>
        public static int LoweredRegions(IGraphPlan plan)
            => plan is Plan p ? p.LoweredRegions : throw new ArgumentException("not a ggml graph plan", nameof(plan));

        /// <summary>Makes <paramref name="weights"/> resident on the device through MiniMax-H3's weight uploader (direct reads
        /// from a registered file mapping, else prefaulted parallel copies: ggml_h3_upload.h), into the device copies a weight
        /// leaf of the same host bytes binds, so the first program that reads them uploads nothing. A weight's host bytes are
        /// its <see cref="IGraphWeight.Data"/>, which must also be its cache key. Returns how many were copied now (0 where the
        /// device reads host memory in place or they were resident already).</summary>
        public int Preload(IReadOnlyList<IGraphWeight> weights, string tag)
        {
            ArgumentNullException.ThrowIfNull(weights);
            var items = new GgmlPreloadWeight[weights.Count];
            for (int i = 0; i < items.Length; i++)
            {
                IGraphWeight w = weights[i] ?? throw new ArgumentException($"weight {i} is null", nameof(weights));
                if (w.Data == IntPtr.Zero || (w.CacheKey != IntPtr.Zero && w.CacheKey != w.Data))
                    throw new ArgumentException($"weight {i} is not keyed by its host bytes", nameof(weights));
                items[i].Data = w.Data;
                items[i].Bytes = w.Bytes;
                items[i].Type = w.Format;
                unsafe
                {
                    items[i].Ne[0] = w.Ne0;
                    items[i].Ne[1] = w.Ne1;
                    items[i].Ne[2] = w.Ne2;
                    items[i].Nd = w.Ne2 > 1 ? 3 : 2;
                }
            }
            unsafe
            {
                int uploaded = 0;
                fixed (GgmlPreloadWeight* p = items)
                    if (GgmlNative.TSGgml_GraphPreloadWeights(p, items.Length, tag, &uploaded) == 0)
                        throw new InvalidOperationException("ggml weight preload failed: " + GgmlBasicOps.LastNativeError());
                return uploaded;
            }
        }

        public void SyncToHost(IntPtr data, long bytes) => GgmlBasicOps.SyncHostBuffer(data, bytes);

        public void Invalidate(IntPtr data) => GgmlBasicOps.InvalidateHostBuffer(data);

        private sealed class Plan : IGraphPlan
        {
            private readonly GgmlGraph _graph;
            private readonly GCHandle[] _pins;

            public Plan(GgmlGraph graph, float[][] outputs, GCHandle[] pins, int servedRegions, int loweredRegions)
            {
                _graph = graph;
                Outputs = outputs;
                _pins = pins;
                ServedRegions = servedRegions;
                LoweredRegions = loweredRegions;
            }

            public int ServedRegions { get; }

            public int LoweredRegions { get; }

            public float[][] Outputs { get; }

            internal GgmlGraph Graph => _graph;

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
