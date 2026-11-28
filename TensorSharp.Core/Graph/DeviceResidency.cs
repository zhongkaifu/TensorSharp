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
    /// <summary>
    /// The owner side of host bytes that backends keep device copies of (D02). A backend may keep a resident copy of a weight
    /// or a state keyed by its host address (the GGML executors' pointer cache: <see cref="GraphCapabilities.BindsHostPointers"/>)
    /// or by a quantized weight's cache key (the CUDA and MLX resident-weight caches), and keeps it after the plan that made
    /// it is gone. Whoever frees, re-pins or rewrites such bytes must drop those copies first, or the next bytes at that
    /// address bind the old copy and compute with stale values. That rule used to live in each model's cleanup code, which
    /// had to know each backend's cache (an executor capability check and <see cref="IGraphExecutor.Invalidate"/> here, a
    /// <c>CudaQuantizedOps</c> or <c>MlxQuantizedOps</c> release there). It lives here now: a backend registers how its
    /// resident weight copies are dropped (<see cref="RegisterWeightReleaser"/>, at module load), and a model releases what
    /// it owns with <see cref="ReleaseWeightCopies"/>, <see cref="ReleaseTensor"/> or a <see cref="ResidentTensorLease"/>.
    /// Nothing here synchronizes the device: the copies are dropped where their owner retires them, as before.
    /// </summary>
    public static class DeviceResidency
    {
        private static readonly ConcurrentDictionary<Type, Action<IAllocator, IntPtr>> s_weightReleasers = new();

        /// <summary>Registers how a backend whose allocator is <paramref name="allocatorType"/> drops the resident copy it keeps of
        /// a quantized weight under its cache key (a backend's module initializer calls this).</summary>
        public static void RegisterWeightReleaser(Type allocatorType, Action<IAllocator, IntPtr> release)
        {
            ArgumentNullException.ThrowIfNull(allocatorType);
            ArgumentNullException.ThrowIfNull(release);
            s_weightReleasers[allocatorType] = release;
        }

        /// <summary>Whether <paramref name="allocator"/>'s backend keeps resident weight copies by cache key.</summary>
        public static bool KeepsWeightCopies(IAllocator? allocator)
            => allocator != null && s_weightReleasers.ContainsKey(allocator.GetType());

        /// <summary>Drops every device copy of a weight before its host blocks are freed: the resident copy
        /// <paramref name="allocator"/>'s backend keeps under <paramref name="cacheKey"/> (where it registered one), and the copy
        /// <paramref name="executor"/> keeps of the blocks at <paramref name="hostData"/>. Without an executor named, the
        /// allocator's own executor is asked when the backend keeps no resident weight cache and that executor binds host
        /// pointers; no executor is made just to ask otherwise.</summary>
        public static void ReleaseWeightCopies(IAllocator? allocator, IntPtr cacheKey, IntPtr hostData, IGraphExecutor? executor = null)
        {
            if (allocator != null && s_weightReleasers.TryGetValue(allocator.GetType(), out var release))
            {
                if (cacheKey != IntPtr.Zero) release(allocator, cacheKey);
                Invalidate(hostData, executor);
                return;
            }
            if (executor != null) Invalidate(hostData, executor);
            else InvalidateBound(hostData, GraphExecutors.For(allocator));
        }

        /// <summary>Tells <paramref name="executor"/> the bytes at <paramref name="data"/> are going (it drops whatever copy it
        /// keeps of them; an executor that keeps none ignores it).</summary>
        public static void Invalidate(IntPtr data, IGraphExecutor? executor)
        {
            if (data != IntPtr.Zero && executor != null)
                executor.Invalidate(data);
        }

        /// <summary>Drops the copies an executor that binds host pointers keeps of the bytes at <paramref name="data"/>.</summary>
        public static void InvalidateBound(IntPtr data, IGraphExecutor? executor)
        {
            if (data != IntPtr.Zero && executor is { Capabilities.BindsHostPointers: true })
                executor.Invalidate(data);
        }

        /// <summary>Drops the device copies of <paramref name="tensor"/>'s bytes, then disposes it: the release of a tensor a
        /// graph bound by its host address. <paramref name="executor"/>, when named, is told; otherwise the tensor's allocator's
        /// executor, where it binds host pointers.</summary>
        public static void ReleaseTensor(Tensor? tensor, IGraphExecutor? executor = null)
        {
            if (tensor == null) return;
            IntPtr data = TensorComputePrimitives.GetStoragePointer(tensor);
            if (executor != null) Invalidate(data, executor);
            else InvalidateBound(data, GraphExecutors.For(tensor.Allocator));
            tensor.Dispose();
        }
    }

    /// <summary>
    /// A tensor whose host bytes graph executors may bind by address, owned for as long as the lease lives: disposing the lease
    /// drops the device copies first and the tensor after (D02's <c>BoundTensorLease</c>), so the owner never orders the two
    /// itself. A view's lease releases only the view: the bytes belong to the owner's lease.
    /// </summary>
    public sealed class ResidentTensorLease : IDisposable
    {
        private Tensor? _tensor;
        private readonly IGraphExecutor? _executor;

        public ResidentTensorLease(Tensor tensor, IGraphExecutor? executor = null)
        {
            _tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
            _executor = executor;
        }

        /// <summary>The leased tensor (null once released).</summary>
        public Tensor? Tensor => _tensor;

        public void Dispose()
        {
            Tensor? tensor = _tensor;
            _tensor = null;
            DeviceResidency.ReleaseTensor(tensor, _executor);
        }
    }
}
