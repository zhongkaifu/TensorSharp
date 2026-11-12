// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Concurrent;

namespace TensorSharp.Graph
{
    /// <summary>What a backend's executor does beyond running a program.</summary>
    public sealed record GraphCapabilities
    {
        /// <summary>Its kernels read row-contiguous strided views directly (no contiguous copies needed).</summary>
        public bool StridedViews { get; init; }
        /// <summary>An op may write its result in place inside a state leaf (<see cref="GraphNodeKind.Alias"/>).</summary>
        public bool InPlaceStateAlias { get; init; }
        /// <summary>Reordering independent work pays off (and is safe) on this device.</summary>
        public bool ConcurrencyReorder { get; init; }
        /// <summary>The bucket a decode step's padded attention window grows by (a window's programs replay).</summary>
        public int DecodeWindowRows { get; init; } = 256;
        /// <summary>Programs may spread over several ranks (<see cref="GraphProgram.TensorParallelRanks"/>).</summary>
        public bool TensorParallel { get; init; }
        /// <summary>Prompt chunks write K/V by row index (no copy into a strided cache view).</summary>
        public bool IndexedCacheWrites { get; init; }
        /// <summary>A prompt chunk over an F16 or block-quantized K/V cache is written by row index and attends the
        /// cache's window padded to <see cref="DecodeWindowRows"/>, as decode does, instead of its own fresh F32 rows:
        /// the device's fast attention needs reduced-precision K/V in whole windows. An F32 cache keeps the fresh rows
        /// (exact, and no faster that way).</summary>
        public bool PaddedCacheAttention { get; init; }
        /// <summary>K/V caches may be block-quantized (Q8_0, Q4_0): views and writes address their blocks.</summary>
        public bool QuantizedKvCache { get; init; }
        /// <summary>It runs <see cref="GraphOp.MoeHostExperts"/>: routed experts computed from host memory at a
        /// pause in the program.</summary>
        public bool HostExperts { get; init; }
        /// <summary>A compiled plan is worth keeping to run again (a replayable program is cached and
        /// looked up by its signature); otherwise every step is recorded and compiled once.</summary>
        public bool Replays { get; init; } = true;
        /// <summary>Leaves are bound by their host bytes (GGML's device caches are keyed by host
        /// address); otherwise by the leaf's object (<see cref="GraphNode.Handle"/>), and a leaf's
        /// <see cref="GraphNode.Data"/> is only an identity.</summary>
        public bool BindsHostPointers { get; init; }
        /// <summary>A leaf bound by its host bytes is computed on through a separate device copy, moved to and from
        /// the host by syncs; false where the device reads the host bytes themselves (Metal's shared buffers, the
        /// CPU), so a host copy of them moves nothing across.</summary>
        public bool SeparateDeviceCopies { get; init; }
    }

    /// <summary>A stable identity for a storage object, for program signatures of executors that bind
    /// leaves by object (taking a device storage's host pointer would sync it).</summary>
    public static class GraphIdentity
    {
        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, object> s_ids = new();
        private static long s_next = 1L << 40;

        public static IntPtr Of(object storage, long offset = 0)
        {
            object id = s_ids.GetValue(storage, _ => System.Threading.Interlocked.Increment(ref s_next));
            return new IntPtr(((long)id << 8) + offset);
        }
    }

    /// <summary>
    /// A backend's graph executor: the plugin through which every model runs. It compiles a
    /// backend-neutral <see cref="GraphProgram"/> into a plan for its devices and keeps the
    /// leaves' bytes coherent between the host and the device.
    /// </summary>
    public interface IGraphExecutor
    {
        GraphCapabilities Capabilities { get; }

        /// <summary>Compile <paramref name="program"/>; throws <see cref="NotSupportedException"/> when it
        /// uses something this backend cannot run.</summary>
        IGraphPlan Compile(GraphProgram program);

        /// <summary>The newest bytes of a state leaf at <paramref name="data"/> are copied back to the host.</summary>
        void SyncToHost(IntPtr data, long bytes);

        /// <summary>The host bytes at <paramref name="data"/> changed (or are being freed): drop device copies.</summary>
        void Invalidate(IntPtr data);
    }

    /// <summary>A compiled program. Outputs are the plan's arrays, rewritten by each run.</summary>
    public interface IGraphPlan : IDisposable
    {
        float[][] Outputs { get; }

        /// <summary>Run with <paramref name="inputs"/>' values (node, host bytes, byte count); false with
        /// <see cref="LastError"/> set on failure.</summary>
        bool Run(ReadOnlySpan<int> inputs, ReadOnlySpan<IntPtr> data, ReadOnlySpan<long> bytes);

        /// <summary>As <see cref="Run(ReadOnlySpan{int}, ReadOnlySpan{IntPtr}, ReadOnlySpan{long})"/>, with an
        /// optional object per input (a <see cref="Tensor"/> an executor binding objects reads directly).</summary>
        bool Run(ReadOnlySpan<int> inputs, ReadOnlySpan<IntPtr> data, ReadOnlySpan<long> bytes, ReadOnlySpan<object> handles)
            => Run(inputs, data, bytes);

        /// <summary>Leaf <paramref name="node"/> is bound to device memory that outlives a run.</summary>
        bool IsLeafResident(int node);

        string LastError { get; }
    }

    /// <summary>Receives a program's entries as they are recorded (node index -1 for entries that are
    /// not nodes), and each input leaf's per-run value right after the leaf.</summary>
    public interface IGraphSink
    {
        void Entry(int node, in GraphNode entry);
        void InputData(int node, IntPtr data, long bytes, object handle);
    }

    /// <summary>
    /// A program that runs as it is recorded: its nodes are evaluated as they arrive (the emitting code
    /// runs inside <see cref="Execute"/>, on the backend's thread), and <see cref="IGraphPlan.Run"/>
    /// then only reads the outputs. For executors whose plans would not replay anyway, so recording
    /// overlaps the device's work instead of preceding it.
    /// </summary>
    public interface IGraphStream : IGraphSink, IGraphPlan
    {
        bool Execute(Func<bool> body);
    }

    /// <summary>An executor that can also run programs as they are recorded.</summary>
    public interface IStreamingGraphExecutor : IGraphExecutor
    {
        /// <summary>A stream for one step. <paramref name="template"/>, when given, is a recorded program of
        /// the same kind: the stream releases each node's tensors after its last reader (it checks that the
        /// step matches the template node by node, and keeps everything to the end where it does not).</summary>
        IGraphStream OpenStream(GraphProgram? template = null);
    }

    /// <summary>One rank's inputs for <see cref="IRankedGraphExecutor.RunRanked"/>: the input nodes with their per-run
    /// host bytes and byte counts.</summary>
    public readonly record struct GraphRunInputs(int[] Nodes, IntPtr[] Data, long[] Bytes);

    /// <summary>
    /// An executor whose compiled plans of one tensor-parallel step run together: plan r, compiled on rank r, runs
    /// with its inputs while every rank pauses at each <see cref="GraphOp.AllReduce"/> to sum the partial in place
    /// (on the devices where the backend has a collective), then continues. For executors that compile whole
    /// programs; a streaming executor reduces as the ranks' nodes arrive instead.
    /// </summary>
    public interface IRankedGraphExecutor : IGraphExecutor
    {
        /// <summary>Whether steps over <paramref name="ranks"/> devices can run here (the devices' collective is up).</summary>
        bool RunsRanks(int ranks);

        /// <summary>Compile <paramref name="program"/> for rank <paramref name="rank"/>'s device.</summary>
        IGraphPlan CompileForRank(GraphProgram program, int rank);

        /// <summary>Run every rank's plan together; false with <paramref name="error"/> on failure.</summary>
        bool RunRanked(IGraphPlan[] plans, GraphRunInputs[] inputs, out string error);
    }

    /// <summary>A plan whose outputs can stay on the device: set <see cref="KeepTensors"/> before a run and
    /// take each output as a tensor after it (no host copy, no wait for the device).</summary>
    public interface IGraphTensorOutputs
    {
        bool KeepTensors { get; set; }

        /// <summary>Output <paramref name="index"/> of the last run (the caller owns it).</summary>
        Tensor TakeOutput(int index);
    }

    /// <summary>Executor plugins, by the allocator type of the backend they serve.</summary>
    public static class GraphExecutors
    {
        private static readonly ConcurrentDictionary<Type, Func<IAllocator, IGraphExecutor>> s_factories = new();

        /// <summary>Register the executor of backends whose allocators are <paramref name="allocatorType"/>
        /// (a backend assembly calls this from its module initializer).</summary>
        public static void Register(Type allocatorType, Func<IAllocator, IGraphExecutor> factory)
            => s_factories[allocatorType ?? throw new ArgumentNullException(nameof(allocatorType))] =
                factory ?? throw new ArgumentNullException(nameof(factory));

        /// <summary>The executor for <paramref name="allocator"/>'s backend, or null when none is registered.</summary>
        public static IGraphExecutor? For(IAllocator? allocator)
        {
            if (allocator == null) return null;
            for (Type? t = allocator.GetType(); t != null; t = t.BaseType)
                if (s_factories.TryGetValue(t, out Func<IAllocator, IGraphExecutor>? factory))
                    return factory(allocator);
            return null;
        }
    }
}
