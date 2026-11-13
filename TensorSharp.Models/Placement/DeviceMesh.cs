// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.Placement
{
    /// <summary>
    /// The devices one model instance runs on: <see cref="Stages"/> whole-layer pipeline
    /// stages, each a tensor-parallel group of <see cref="TensorParallel"/> devices. Device
    /// (stage s, rank r) is global device s*T + r. Single-device is 1 x 1, <c>--tp T</c> is
    /// 1 x T, <c>--layer-split P</c> is P x 1 and the hybrid is P x T.
    /// </summary>
    internal sealed class DeviceMesh
    {
        private readonly Func<int, IAllocator> _deviceAllocator;
        private readonly ITensorParallelGroup _devices;
        private readonly ITensorParallelGroup[] _stageGroups;

        /// <param name="stages">P.</param>
        /// <param name="tensorParallel">T.</param>
        /// <param name="deviceAllocator">Allocator of global device s*T + r.</param>
        /// <param name="devices">The group spanning all P*T devices when T &gt; 1 (it owns them and
        /// its per-rank pinning); null when every stage is a single device.</param>
        /// <param name="plan">Layer to stage assignment; null for a single stage.</param>
        public DeviceMesh(int stages, int tensorParallel, Func<int, IAllocator> deviceAllocator,
            ITensorParallelGroup devices, LayerSplitPlan plan)
        {
            if (stages < 1) throw new ArgumentOutOfRangeException(nameof(stages));
            if (tensorParallel < 1) throw new ArgumentOutOfRangeException(nameof(tensorParallel));
            if (tensorParallel > 1 && devices == null)
                throw new ArgumentException("Tensor parallelism needs a device group.", nameof(devices));
            if (stages > 1 && plan == null)
                throw new ArgumentException("Several stages need a layer plan.", nameof(plan));
            Stages = stages;
            TensorParallel = tensorParallel;
            Plan = plan;
            _deviceAllocator = deviceAllocator ?? throw new ArgumentNullException(nameof(deviceAllocator));
            _devices = devices;
            if (tensorParallel > 1)
            {
                _stageGroups = new ITensorParallelGroup[stages];
                for (int s = 0; s < stages; s++)
                    _stageGroups[s] = stages == 1 ? devices : new StageGroup(devices, s * tensorParallel, tensorParallel);
            }
        }

        public int Stages { get; }
        public int TensorParallel { get; }
        public LayerSplitPlan Plan { get; }
        public bool IsSingleDevice => Stages == 1 && TensorParallel == 1;
        public int DeviceCount => Stages * TensorParallel;

        public int StageOf(int layer) => Plan == null ? 0 : Plan.LocalRank(layer);

        public int GlobalDevice(int stage, int rank) => stage * TensorParallel + rank;

        public IAllocator Allocator(int stage, int rank) => _deviceAllocator(GlobalDevice(stage, rank));

        /// <summary>The tensor-parallel group of <paramref name="stage"/>; null when T == 1.
        /// With one stage it is the full device group itself (its native collectives, and its
        /// cross-node reduction for distributed TP); with several it is a sub-group.</summary>
        public ITensorParallelGroup Group(int stage) => _stageGroups?[stage];

        /// <summary>Run <paramref name="body"/> for every rank of <paramref name="stage"/>, with the
        /// group's concurrency and device pinning. Inline for a single-device stage.</summary>
        public void RunPerRank(int stage, Action<int> body)
        {
            if (TensorParallel == 1) { body(0); return; }
            _stageGroups[stage].RunPerRank(body);
        }

        /// <summary>Sum the stage's per-rank partials in place (every rank gets the total).</summary>
        public void AllReduce(int stage, Tensor[] partials)
        {
            if (TensorParallel == 1) return;
            _stageGroups[stage].AllReduce(partials);
        }

        public string Describe() => IsSingleDevice
            ? "1 device"
            : Stages == 1
                ? $"tensor parallel {TensorParallel}"
                : TensorParallel == 1
                    ? $"{Stages} whole-layer stages"
                    : $"{Stages} whole-layer stages x tensor parallel {TensorParallel} = {DeviceCount} devices";

        /// <summary>
        /// One stage's ranks inside a larger local device group. Work runs through the parent's
        /// <see cref="ITensorParallelGroup.RunPerRank"/>, so each rank keeps the parent's device
        /// pinning and concurrency; ranks of other stages return immediately. The reduction is an
        /// exact host-staged sum over the stage's tensors: the parent's native collective spans all
        /// of its devices and cannot reduce a subset (a native sub-group collective is Phase 2 work,
        /// see docs/design/unified-execution.md).
        /// </summary>
        private sealed class StageGroup(ITensorParallelGroup parent, int first, int degree) : ITensorParallelGroup
        {
            public int Degree => degree;
            public bool IsActive => degree > 1;
            public int GlobalDegree => degree;
            public int GlobalRankOffset => 0;
            public int NodeCount => 1;

            public IAllocator GetAllocator(int rank) => parent.GetAllocator(first + rank);

            public void RunPerRank(Action<int> body)
                => parent.RunPerRank(global =>
                {
                    int local = global - first;
                    if (local >= 0 && local < degree)
                        body(local);
                });

            public void AllReduce(Tensor[] tensors)
            {
                if (tensors == null || tensors.Length != degree)
                    throw new ArgumentException($"Expected {degree} tensors, got {tensors?.Length ?? 0}.", nameof(tensors));
                parent.Synchronize();
                SimulatedDeviceGroup.HostAllReduce(tensors);
            }

            public void Synchronize() => parent.Synchronize();
            public void Barrier() { }
            public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException("A stage group has no worker nodes.");
            public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException("A stage group has no worker nodes.");
            public void Dispose() { }   // the parent owns the devices
        }
    }
}
