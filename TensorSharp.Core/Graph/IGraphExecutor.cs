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
        /// <summary>Leaves are bound by their host bytes (GGML's device caches are keyed by host
        /// address); otherwise by the leaf's object (<see cref="GraphNode.Handle"/>), and a leaf's
        /// <see cref="GraphNode.Data"/> is only an identity.</summary>
        public bool BindsHostPointers { get; init; }
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
        public static IGraphExecutor For(IAllocator allocator)
        {
            if (allocator == null) return null;
            for (Type t = allocator.GetType(); t != null; t = t.BaseType)
                if (s_factories.TryGetValue(t, out var factory))
                    return factory(allocator);
            return null;
        }
    }
}
