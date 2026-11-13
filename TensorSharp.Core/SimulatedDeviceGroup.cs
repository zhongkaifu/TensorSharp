// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;

using TensorSharp.Cpu;

namespace TensorSharp
{
    /// <summary>
    /// A local device group whose "devices" are independent host-memory allocators, with an
    /// exact host-side AllReduce. It exercises every placement path of the runtime
    /// (tensor parallelism, layer split and their hybrid) on any machine, so the conformance
    /// tests can prove a mechanism without GPUs. It is a FUNCTIONAL device model only: it is
    /// never evidence of multi-device performance, and the loader never selects it.
    /// </summary>
    public sealed class SimulatedDeviceGroup : ITensorParallelGroup
    {
        private readonly CpuAllocator[] _allocators;

        public SimulatedDeviceGroup(int degree)
        {
            if (degree < 1) throw new ArgumentOutOfRangeException(nameof(degree));
            _allocators = Enumerable.Range(0, degree).Select(_ => new CpuAllocator(BlasEnum.DotNet)).ToArray();
        }

        public int Degree => _allocators.Length;
        public bool IsActive => Degree > 1;
        public int GlobalDegree => Degree;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public bool IsDisposed { get; private set; }

        /// <summary>Number of AllReduce calls served, for tests that assert where reductions happen.</summary>
        public int AllReduceCount { get; private set; }

        public IAllocator GetAllocator(int rank) => _allocators[rank];

        public void AllReduce(Tensor[] tensors)
        {
            if (tensors == null || tensors.Length != Degree)
                throw new ArgumentException($"Expected {Degree} tensors, got {tensors?.Length ?? 0}.", nameof(tensors));
            AllReduceCount++;
            HostAllReduce(tensors);
        }

        /// <summary>Exact element-wise sum through host memory, left in every tensor. Used by
        /// any group (or sub-group) that has no faster collective for its device set.</summary>
        public static void HostAllReduce(Tensor[] tensors)
        {
            if (tensors.Length <= 1) return;
            long count = tensors[0].ElementCount();
            for (int r = 1; r < tensors.Length; r++)
            {
                if (tensors[r].ElementCount() != count)
                    throw new ArgumentException("AllReduce requires identically sized tensors on every rank.", nameof(tensors));
            }
            float[] sum = tensors[0].GetElementsAsFloat((int)count);
            for (int r = 1; r < tensors.Length; r++)
            {
                float[] part = tensors[r].GetElementsAsFloat((int)count);
                for (int i = 0; i < sum.Length; i++)
                    sum[i] += part[i];
            }
            foreach (Tensor tensor in tensors)
            {
                tensor.SetElementsAsFloat(sum);
                // Device-mirrored storages (direct CUDA) re-upload the host write.
                tensor.EnsureDeviceCurrent();
            }
        }

        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException("A simulated device group has no worker nodes.");
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException("A simulated device group has no worker nodes.");
        public void Dispose() => IsDisposed = true;
    }
}
